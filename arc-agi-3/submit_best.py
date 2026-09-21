"""One-shot, score-aware submission for ARC-AGI-3 (+ the AGI-2 DSL kernel).

Safe to run any number of times: it checks the competition's own
`numAllowedNow` counter before doing anything, so it can never double-spend
a daily submission slot. Designed for scheduled/cloud sessions that suspend
when idle -- it performs one decision and exits (use --watch for the old
sleep-and-retry behaviour on an always-on box).

Candidate policy (ARC-AGI-3 slot):
  * Any COMPLETE candidate whose kernel output's summary.txt reports
    "mean score" >= STRONG_MEAN  -> submit immediately (beats the 27B floor).
  * Otherwise, once the clock passes LATE_HHMM UTC -> submit the completed
    candidate with the highest verified mean >= MIN_MEAN.
  * Otherwise, once the clock passes FINAL_HHMM UTC -> submit the Forge
    fallback (scores ~0.08; better than leaving the slot unused).
  * Otherwise -> do nothing; a later run can still pick up a better kernel.

Usage:
  .venv/bin/python submit_best.py             # one decision, then exit
  .venv/bin/python submit_best.py --watch     # retry every 15min for 8h
  .venv/bin/python submit_best.py --dry-run   # print the decision only
"""
import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kaggle.api.kaggle_api_extended import KaggleApi

LOG = logging.getLogger("submit_best")

VERMAP_FILE = HERE / "kernel_versions.json"

# (kernel slug, pinned kernel_version or None -> map file, submission message).
# Ordered by preference when verified means tie; highest mean wins overall.
CANDIDATES = [
    ("takumuhata/arc3-duck-anim-flashnext", None,
     "TAAF anim-aware solver + Qwen3.8 Flash-Next NVFP4 MTP (hybrid)"),
    ("takumuhata/arc3-duck-qwen3-8-flash-next-nvfp4-mtp", None,
     "Duck Qwen3.8 Flash-Next NVFP4 MTP tuned (public25 profile)"),
    ("takumuhata/arc3-duck-qwen3-8-27b", 2,
     "TAAF duck + Qwen3.6-27B-FP8 + capped-output solver patch"),
]
# Already-measured means (avoids re-downloading kernel output). Consulted
# ONLY when the live download yields nothing -- e.g. a newer kernel version is
# QUEUED/RUNNING so kernels_output serves that empty tree, hiding the last
# completed run. A fresh completed run's own summary/score.json always wins.
KNOWN_MEAN = {
    "takumuhata/arc3-duck-qwen3-8-27b": 4.79,  # v1 run; v2 (patched agent) in flight
}
STRONG_MEAN = 5.5   # submit early only if a candidate clearly beats 27B (4.79)
MIN_MEAN = 4.0      # late in the day accept anything >= this
LATE_HHMM = (22, 0)   # UTC: after this, best verified >= MIN_MEAN goes out
FINAL_HHMM = (23, 30)  # UTC: after this, fall back to Forge
RETRY_S = 900
WATCH_ATTEMPTS = 32


def _hhmm(s):
    h, m = s.split(":")
    return (int(h), int(m))

FALLBACK_JOB = dict(competition="arc-prize-2026-arc-agi-3",
                    kernel="takumuhata/forge-pathfinder-bfs-agent",
                    file_name="submission.parquet",
                    message="Forge v7: dynamic valid-actions + embedded solutions + early stop")

AGI2_JOB = dict(competition="arc-prize-2026-arc-agi-2",
                kernel="takumuhata/arc-agi2-lb33-perfpatch-dsl",
                kernel_version=1,
                file_name="submission.json",
                message="perfpatch LB33.89 + DSL symbolic injection")

api = KaggleApi()
api.authenticate()


def log(msg):
    LOG.info(msg)
    print(f"{datetime.now(timezone.utc):%H:%M:%S} {msg}", flush=True)


def allowed_now(competition):
    """Competition's own remaining-submissions counter. None on lookup failure."""
    try:
        lim = api.competition_get_submission_limits(competition)
        for attr in ("num_allowed_now", "numAllowedNow"):
            v = getattr(lim, attr, None)
            if v is not None:
                return int(v)
        m = re.search(r'"numAllowedNow":\s*(\d+)', str(lim))
        if m:
            return int(m.group(1))
        log(f"{competition}: limits object missing numAllowedNow: {str(lim)[:140]}")
        return None
    except Exception as e:
        log(f"{competition}: limits lookup failed {str(e)[:140]}")
        return None


def kernel_status(slug):
    try:
        st = api.kernels_status(slug)
        return getattr(st, "status", None) or str(st)
    except Exception as e:
        log(f"{slug}: status failed {str(e)[:120]}")
        return None


def kernel_complete(slug):
    status = kernel_status(slug)
    return status is not None and "COMPLETE" in str(status)


def mapped_version(slug):
    try:
        vermap = json.loads(VERMAP_FILE.read_text())
        v = vermap.get(slug)
        return int(v) if v is not None else None
    except Exception:
        return None


def _mean_from_score_json(path):
    """Mean of per-game 'score' values in a score.json-shaped file."""
    try:
        d = json.loads(Path(path).read_text(errors="replace"))
        if isinstance(d.get("score"), (int, float)):
            return float(d["score"])
        vals = [float(g["score"]) for g in (d.get("games") or {}).values()
                if "score" in g]
        return sum(vals) / len(vals) if vals else None
    except Exception:
        return None


def verified_mean(slug):
    """Fetch the kernel's score summary (summary.txt or score.json) and parse
    the mean public-eval score. None on failure."""
    try:
        with tempfile.TemporaryDirectory() as td:
            try:
                api.kernels_output(slug, td, file_pattern="summary.txt")
            except Exception:
                pass
            if not list(Path(td).rglob("summary.txt")):
                try:
                    api.kernels_output(slug, td, file_pattern="score.json")
                except Exception:
                    pass
            # The kernel .log also lands here and prints a running
            # "mean score:" line per checkpoint -- take the LAST match in
            # any .txt, which is the final value.
            for p in Path(td).rglob("*.txt"):
                matches = re.findall(
                    r"mean score:\s*([0-9.]+)", p.read_text(errors="replace")
                )
                if matches:
                    return float(matches[-1])
            for p in Path(td).rglob("score.json"):
                mean = _mean_from_score_json(p)
                if mean is not None:
                    return mean
    except Exception as e:
        log(f"{slug}: mean fetch failed {str(e)[:120]}")
    # Output of the last completed run is hidden while a newer version is
    # queued/running; fall back to the last hand-verified mean.
    if slug in KNOWN_MEAN:
        return KNOWN_MEAN[slug]
    return None


def submit(job):
    """Try the mapped/pinned version first, then latest (None)."""
    ver = job.get("kernel_version") or mapped_version(job["kernel"])
    try:
        return api.competition_submit_code(
            file_name=job["file_name"], message=job["message"],
            competition=job["competition"], kernel=job["kernel"],
            kernel_version=ver)
    except Exception:
        if ver is None:
            raise
        log(f"{job['kernel']} v{ver} rejected; retrying with latest version")
        return api.competition_submit_code(
            file_name=job["file_name"], message=job["message"],
            competition=job["competition"], kernel=job["kernel"],
            kernel_version=None)


def pick_candidate():
    """Return (mean, slug, version, message) of best completed candidate.

    A kernel stays eligible while a newer version is queued/running:
    ``kernels_output`` returns the last COMPLETED run's artifacts, so
    ``verified_mean`` still reflects proven code. The submitted version is
    pinned to whichever version produced that output: when the kernel's
    current status is COMPLETE, the latest pushed version ran to completion
    and owns the output; while QUEUED/RUNNING, the output belongs to the
    previous version, so submit one behind the vermap/pin.
    """
    best = None
    for slug, ver, msg in CANDIDATES:
        status = kernel_status(slug)
        if status is None:
            log(f"{slug}: status lookup failed")
            continue
        mean = verified_mean(slug)
        log(f"{slug}: status={status} mean={mean}")
        if mean is None:
            continue
        eff_ver = ver or mapped_version(slug)
        if "COMPLETE" not in str(status) and eff_ver and eff_ver > 1:
            eff_ver -= 1  # in-flight version hasn't produced output yet
        if best is None or mean > best[0]:
            best = (mean, slug, eff_ver, msg)
    return best


def attempt(dry_run=False, allow_fallback=True, force_final=False):
    """One submission decision for both competitions. Returns dict of outcomes."""
    out = {"agi3": None, "agi2": None}

    if allowed_now(AGI2_JOB["competition"]):
        job = dict(AGI2_JOB)
        log(f"AGI-2 slot open -> {job['kernel']}")
        if not dry_run:
            try:
                r = submit(job)
                log(f"AGI-2 SUBMIT OK {job['kernel']}: {r}")
                out["agi2"] = job["kernel"]
            except Exception as e:
                log(f"AGI-2 submit failed: {str(e)[:200]}")
        else:
            out["agi2"] = "dry-run"

    if not allowed_now(FALLBACK_JOB["competition"]):
        log("AGI-3 slot not open (already used today or lookup failed)")
        return out

    best = pick_candidate()
    now = datetime.now(timezone.utc)
    late = (now.hour, now.minute) >= LATE_HHMM
    final = force_final or (now.hour, now.minute) >= FINAL_HHMM

    chosen = None
    if best is not None:
        mean, slug, ver, msg = best
        if mean >= STRONG_MEAN or (late and mean >= MIN_MEAN) or final:
            # At the final window any verified duck candidate beats Forge (~0.08).
            chosen = dict(competition="arc-prize-2026-arc-agi-3",
                          kernel=slug, file_name="submission.parquet",
                          kernel_version=ver or mapped_version(slug),
                          message=f"{msg} [mean {mean}]")
        else:
            log(f"holding {slug} mean {mean}: below STRONG {STRONG_MEAN} "
                f"and before {LATE_HHMM[0]:02d}:{LATE_HHMM[1]:02d} UTC")
    if chosen is None and allow_fallback:
        if final:
            chosen = dict(FALLBACK_JOB)
            log("final window -> Forge fallback")
        else:
            log(f"no verified candidate; holding fallback until "
                f"{FINAL_HHMM[0]:02d}:{FINAL_HHMM[1]:02d} UTC")

    if chosen is None:
        log("AGI-3: nothing to submit this run")
    else:
        log(f"AGI-3 -> {chosen['kernel']} :: {chosen['message']}")
        if not dry_run:
            try:
                r = submit(chosen)
                log(f"AGI-3 SUBMIT OK {chosen['kernel']}: {r}")
                out["agi3"] = chosen["kernel"]
            except Exception as e:
                log(f"AGI-3 submit failed: {str(e)[:200]}")
        else:
            out["agi3"] = "dry-run"
    return out


def main():
    global MIN_MEAN, STRONG_MEAN, LATE_HHMM, FINAL_HHMM
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true",
                    help="retry every 15min until both submissions land (8h max)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-fallback", action="store_true",
                    help="never submit the Forge fallback")
    ap.add_argument("--min-mean", type=float, help=f"late-window floor (default {MIN_MEAN})")
    ap.add_argument("--strong-mean", type=float, help=f"early-submit floor (default {STRONG_MEAN})")
    ap.add_argument("--late", metavar="HH:MM", help=f"UTC late window start (default {LATE_HHMM[0]:02d}:{LATE_HHMM[1]:02d})")
    ap.add_argument("--final", metavar="HH:MM", help=f"UTC fallback window start (default {FINAL_HHMM[0]:02d}:{FINAL_HHMM[1]:02d})")
    args = ap.parse_args()

    if args.min_mean is not None:
        MIN_MEAN = args.min_mean
    if args.strong_mean is not None:
        STRONG_MEAN = args.strong_mean
    if args.late:
        LATE_HHMM = _hhmm(args.late)
    if args.final:
        FINAL_HHMM = _hhmm(args.final)

    logging.basicConfig(filename=str(HERE / "submit_best.log"), level=logging.INFO,
                        format="%(asctime)s %(message)s")

    if not args.watch:
        attempt(dry_run=args.dry_run, allow_fallback=not args.no_fallback)
        return

    done = {"agi3": False, "agi2": False}
    for i in range(WATCH_ATTEMPTS):
        # On the last two attempts treat as the final window so a watcher that
        # is about to give up still lands the best verified candidate/fallback.
        force_final = i >= WATCH_ATTEMPTS - 2
        log(f"--- watch attempt {i + 1}/{WATCH_ATTEMPTS}")
        out = attempt(dry_run=args.dry_run, allow_fallback=not args.no_fallback,
                      force_final=force_final)
        for k in done:
            done[k] = done[k] or bool(out[k])
        if all(done.values()):
            break
        time.sleep(RETRY_S)
    log("watch done")


if __name__ == "__main__":
    main()
