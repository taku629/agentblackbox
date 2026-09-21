"""Submit the best verified ARC-AGI-3 kernel once the daily limit resets
(~00:00 UTC; last OCEAN submission went through at 00:14 UTC). Starts at
00:20 UTC and retries every 15 min for up to 8h.

Candidate selection: for each priority kernel that is COMPLETE, download its
summary.txt and parse the public-eval mean score. Submit the completed
candidate with the highest verified mean (27B's 4.79 is hardcoded as the
floor since it is already proven). Falls back to Forge if nothing scores.

Also resubmits the AGI-2 DSL kernel (harmless duplicate if today's pending
run already scored; saves it if it errored).

Logs to watch_submit2.log."""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import logging
from datetime import datetime, timezone, timedelta

sys.path.insert(0, "/home/takumu/kaggle/arc-agi-3")
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(filename="/home/takumu/kaggle/arc-agi-3/watch_submit2.log",
                    level=logging.INFO, format="%(asctime)s %(message)s",
                    force=True)
api = KaggleApi()
api.authenticate()

WAKE_HHMM = (0, 20)
RETRY_S = 900
ATTEMPTS = 32
VERMAP_FILE = "/home/takumu/kaggle/arc-agi-3/kernel_versions.json"

# (kernel slug, kernel_version or None->latest, message). Ordered by
# preference when scores tie; highest verified mean wins overall.
CANDIDATES = [
    ("takumuhata/arc3-duck-anim-flashnext", None,
     "TAAF anim-aware solver + Qwen3.8 Flash-Next NVFP4 MTP (hybrid)"),
    ("takumuhata/arc3-duck-qwen3-8-flash-next-nvfp4-mtp", None,
     "Duck Qwen3.8 Flash-Next NVFP4 MTP tuned (public25 profile)"),
    ("takumuhata/arc3-duck-flash-next-nvfp4-mtp-b", None,
     "Duck Qwen3.8 Flash-Next NVFP4 MTP tuned B"),
    ("takumuhata/arc3-duck-qwen3-8-27b", 1,
     "TAAF duck + Qwen3.6-27B-FP8 on RTX6000 (public eval mean 4.79)"),
]
# Known means where already measured (avoids re-download when unavailable).
KNOWN_MEAN = {"takumuhata/arc3-duck-qwen3-8-27b": 4.79}
MIN_MEAN = 4.0  # only submit a candidate if its verified mean beats this

AGI2_JOB = dict(competition="arc-prize-2026-arc-agi-2",
                kernel="takumuhata/arc-agi2-lb33-perfpatch-dsl",
                kernel_version=1,
                file_name="submission.json",
                message="perfpatch LB33.89 + DSL symbolic injection",
                done=False)

FORGE_JOB = dict(competition="arc-prize-2026-arc-agi-3",
                 kernel="takumuhata/forge-pathfinder-bfs-agent",
                 file_name="submission.parquet",
                 message="Forge v7: dynamic valid-actions + embedded solutions + early stop",
                 done=False)


def kernel_status(slug):
    try:
        out = subprocess.run(["kaggle", "kernels", "status", slug],
                             capture_output=True, text=True, timeout=120).stdout
        return out
    except Exception:
        return ""


def latest_version(slug):
    try:
        vermap = json.load(open(VERMAP_FILE))
        return int(vermap[slug])
    except Exception:
        return None


def verified_mean(slug):
    """Fetch the kernel's summary.txt and parse 'mean score'. None on failure."""
    if slug in KNOWN_MEAN:
        return KNOWN_MEAN[slug]
    try:
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(
                ["kaggle", "kernels", "output", slug, "-p", td,
                 "--file-pattern", "summary.txt"],
                capture_output=True, text=True, timeout=600)
            for f in os.listdir(td):
                if f.endswith(".txt"):
                    m = re.search(r"mean score:\s*([0-9.]+)",
                                  open(os.path.join(td, f), errors="replace").read())
                    if m:
                        return float(m.group(1))
    except Exception as e:
        logging.info(f"{slug}: mean fetch failed {str(e)[:120]}")
    return None


def submit(j, ver=None):
    return api.competition_submit_code(
        file_name=j["file_name"],
        message=j["message"],
        competition=j["competition"],
        kernel=j["kernel"],
        kernel_version=ver or j.get("kernel_version") or latest_version(j["kernel"]))


now = datetime.now(timezone.utc)
target = now.replace(hour=WAKE_HHMM[0], minute=WAKE_HHMM[1], second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
wait = max(0, (target - now).total_seconds())
logging.info(f"sleeping {wait/3600:.1f}h until {target}")
time.sleep(wait)

agi3_done = False
failed = set()
for attempt in range(ATTEMPTS):
    try:
        api.authenticate()
    except Exception as e:
        logging.info(f"re-auth failed (attempt {attempt+1}): {str(e)[:150]}")

    if not agi3_done:
        # Pick the completed candidate with the highest verified mean.
        best = None
        for slug, ver, msg in CANDIDATES:
            if slug in failed:
                continue
            st = kernel_status(slug)
            if "COMPLETE" not in st:
                logging.info(f"{slug}: {st.strip()[:60] or 'unknown'}")
                continue
            mean = verified_mean(slug)
            logging.info(f"{slug}: COMPLETE, mean={mean}")
            if mean is not None and mean >= MIN_MEAN:
                if best is None or mean > best[0]:
                    best = (mean, slug, ver, msg)
        if best is not None:
            mean, slug, ver, msg = best
            try:
                r = api.competition_submit_code(
                    file_name="submission.parquet", message=f"{msg} [mean {mean}]",
                    competition="arc-prize-2026-arc-agi-3",
                    kernel=slug,
                    kernel_version=ver or latest_version(slug))
                logging.info(f"SUBMIT OK {slug} (mean {mean}): {r}")
                agi3_done = True
            except Exception as e:
                err = str(e)[:200]
                logging.info(f"{slug} submit failed (attempt {attempt+1}): {err}")
                # Limit-not-reset errors (400) should retry the same candidate;
                # but a version/validation failure means fall to the next one.
                # Since both surface as 400, give the top candidate a couple of
                # attempts before excluding it — tracked via failed on 3rd strike.
                strikes = sum(1 for s in failed if s.startswith(slug + "#"))
                failed.add(f"{slug}#{strikes}")
                if strikes >= 2:
                    failed.add(slug)

    if not AGI2_JOB["done"]:
        try:
            r = submit(AGI2_JOB)
            logging.info(f"SUBMIT OK {AGI2_JOB['kernel']} v{AGI2_JOB['kernel_version']}: {r}")
            AGI2_JOB["done"] = True
        except Exception as e:
            logging.info(f"{AGI2_JOB['kernel']} submit failed (attempt {attempt+1}): {str(e)[:200]}")

    # Forge only after all candidates were evaluated — never let it win the
    # slot over a viable LLM kernel.
    if not agi3_done and attempt >= 4:
        try:
            r = submit(FORGE_JOB)
            logging.info(f"SUBMIT OK {FORGE_JOB['kernel']}: {r}")
            agi3_done = True
        except Exception as e:
            logging.info(f"{FORGE_JOB['kernel']} submit failed (attempt {attempt+1}): {str(e)[:200]}")

    if agi3_done and AGI2_JOB["done"]:
        break
    time.sleep(RETRY_S)
logging.info("watcher done")
