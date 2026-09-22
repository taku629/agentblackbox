# Cloud Migration Guide

Repo: `taku629/arc-prize-2026-agent-work` (private). Everything needed to
continue the ARC Prize 2026 work in Devin Cloud is here.

## Layout

- `arc-agi-3/` — OCEAN agent (`agent_core.py`), local eval (`run_local.py`),
  submission watcher (`watch_and_submit.py`), dev games (`environment_files/`),
  framework (`ARC-AGI-3-Agents/`), cover image (`cover_dual.png`)
- `ARC-AGI-3-Kaggle-Starter/` — kernel port (`agent/my_agent.py` = OceanCore),
  builder (`scripts/build_notebook.py`), built kernel (`notebooks/submission.ipynb`)
- `arc-prize-2026-paper-track/` — `writeup.md` (1496 words) + `SUBMIT_INSTRUCTIONS.md`
- `arc-agi-2/` — DSL/perfpatch solver (dev/, ~33.9 public-score kernel)
- `submit_ag*/` — all kernel dirs (ipynb + metadata): flash-next x2, hybrid,
  duck 27B/T4 variants, BFS/Forge, AGI-2 DSL/nvarc/perfpatch
- `bundle_hybrid/` + `taaf_src/` — anim-aware solver + Flash-Next NVFP4
  serving stack source (also live as dataset `takumuhata/taaf-anim-flashnext-bundle`)
- `refs/` — downloaded top public kernels for reference
- Fallback full-workspace tarball: private dataset
  `takumuhata/arc3-workspace-migration` (`kaggle datasets download`)

## Secrets

Org secrets: `KAGGLE_ACCESS_TOKEN`, `KAGGLE_CREDENTIALS_JSON`,
`KAGGLE_CREDENTIALS_JSON_20260921` (newest — prefer this one, its
refresh_token is freshest).
In the cloud session, inject them, then restore credential files:

```bash
mkdir -p ~/.kaggle
printf '%s' "$KAGGLE_ACCESS_TOKEN"    > ~/.kaggle/access_token
printf '%s' "$KAGGLE_CREDENTIALS_JSON" > ~/.kaggle/credentials.json
chmod 600 ~/.kaggle/*
```

## Setup (also encoded in the repo's DRS blueprint)

```bash
cd arc-agi-3
python3 -m venv .venv
.venv/bin/pip install kaggle numpy arc-agi arcengine
```

## Common tasks

```bash
# local eval (full suite ~30-60 min)
cd arc-agi-3 && .venv/bin/python run_local.py all 8000
# single game
.venv/bin/python run_local.py ar25 8000

# check submissions / scores
.venv/bin/kaggle competitions submissions arc-prize-2026-arc-agi-3

# rebuild + push kernel after editing agent_core.py / my_agent.py
cd ../ARC-AGI-3-Kaggle-Starter
python3 -m venv .venv && .venv/bin/pip install kaggle nbformat
.venv/bin/python scripts/build_notebook.py
.venv/bin/kaggle kernels push -p notebooks

# IMPORTANT for cloud: sessions suspend when idle, so a long-sleep watcher
# is unreliable. Prefer one-shot, idempotent submission:
cd arc-agi-3 && .venv/bin/python submit_best.py            # one decision, exits
cd arc-agi-3 && .venv/bin/python submit_best.py --dry-run  # print decision only
cd arc-agi-3 && .venv/bin/python push_hybrid_if_missing.py # push anim-hybrid if a slot is free

# submit_best.py checks the competition's own numAllowedNow counter, so it is
# safe to run repeatedly. Policy: a verified mean >= 5.5 submits immediately;
# >= 4.0 submits after 22:00 UTC; Forge fallback goes out after 23:30 UTC.

# local-machine alternative: retry loop (fine on an always-on box)
nohup .venv/bin/python submit_best.py --watch >> submit_best.log 2>&1 &

# poll a submission ref until it leaves PENDING
./poll_sub.sh <ref>   # run in bg
```

## Current state (2026-09-22, evening UTC)

- AGI-3 today: **anim-flashnext submitted 00:25 UTC** (ref `56446903`) —
  its public-eval mean was **8.21**, the automation picked it per policy
  (>= STRONG_MEAN). Hidden-run public score: **2.60** (big public→hidden
  drop as expected — hidden set is different/harder; still 18x the OCEAN
  0.14 from yesterday).
- AGI-2: DSL v1 (ref `56414919`) COMPLETE, **public 29.72**.
- Verified AGI-3 candidate means (public-25 eval): anim-flashnext **8.21**,
  27b v2 **4.97** (v1 was 4.79 — ~±0.2 draw variance), twin NVFP4 4.63.
  anim remains the runaway leader — automation will re-submit it tomorrow
  unless something beats 8.21.
- GPU quota: **weekly 30h exhausted** — `kernels push` now fails with
  "Maximum weekly GPU quota". `arc3-duck-qwen-27b-patched` (27B x patched
  solver) is staged in `submit_ag3_duck_patched/` and will auto-push when
  quota/slots free (`push_hybrid_if_missing.py` retries every automation run).
- OceanCore local baseline (`run_local.py all 3000`, 25 dev games): mean 0.15%
  — matches the 0.14 LB score. Fallback-only; do not rely on it for placement.
- Batch GPU slots are now both free (twin + anim + 27b v2 all COMPLETE);
  further pushes are limited by the weekly 30h quota, not the 2-slot cap.
  The duplicate twin `arc3-duck-flash-next-nvfp4-mtp-b` was deleted earlier
  to free its slot. Re-push it from a directory whose metadata carries that
  id if ever needed again.
- The two kernels differ on TWO axes, not one (corrects earlier "A/B" claim):
  - twin uses keithtyser's baseline bundle: `tool_agent.py` unpatched AND
    **no `animation()` tool** in the python sandbox
  - anim-flashnext uses `takumuhata/taaf-anim-flashnext-bundle`: patched
    `tool_agent.py` — a `finish_reason=="length"` nudge-to-emit retry and
    progressive history shrinking after ≥2 consecutive request failures —
    **plus the `animation()` sandbox tool** (compact diff timeline of frames
    produced by the last action).
  - NOTE: `taaf_setup_env.json` is identical in both runs and sets
    `LOCAL_ANALYZER_MAX_OUTPUT=0` — the output cap is OFF in both; anim's
    win is driven by animation()+nudge+shrink, not the cap.
- Result matrix (public-25 mean): twin 4.63 (no anim, no patch, flash-next) ·
  27b 4.79/4.97 (anim, no patch, 27B) · **anim 8.21 (anim+patch, flash-next (125B-MoE, ~6B active))**.
  anim beats twin on 14 games / loses 5 / ties 6 — it rescued ALL 8 of
  twin's zero-score games (cd82, cn04, dc22, ka59, r11l, sk48, sp80, tn36).
  `animation()` is heavily used: 120–150 calls/game in anim AND in the 27B
  kernel (jakobbrggen bundle has it too) — so anim-vs-27b mostly isolates
  the patch + model speed, not the tool. The staged 27b-patched kernel
  (27B × anim × patch) is the decisive comparison.
- Twin result (COMPLETE ~18:40 UTC): **public mean 4.63** vs 27B's 4.79 —
  roughly a wash, but on a different per-game draw: 121 turns/game avg
  (~2.4x the 27B's ~50), 595 tokens/turn. Wins: lp85 1.82→25.04,
  vc33 8.99→17.88, re86 6.83→16.67. Regressions to zero: r11l, cd82, cn04,
  dc22, sb26 (was 27.78!). Faster turns buy breadth, not depth.
- NOTE: flash-next kernels write `score.json` (games.*.score), not
  `summary.txt` — `submit_best.py`'s verified_mean() handles both now.
- 27B transcript findings (re-downloaded to `~/kout/duck27b`, 129MB):
  only ~50 turns/game at ~144 s/turn under 25-way concurrency; losing games
  correlate with long deliberation (ka59: ~84k thinking tokens over 35 turns,
  single blocks up to ~10k tokens) vs winning games with many short turns
  (sb26: 77 turns, max block ~3k tokens). Timeouts were symptomatic (1–5 per
  game), not the root cause — throughput per turn is.
- `arc3-duck-anim-flashnext` push used to fail on a title/slug mismatch
  (409 Conflict); title fixed to slugify correctly — `push_hybrid_if_missing.py`
  pushes it whenever it is missing and a GPU slot is free.
- Devin automation (pending approval) fires ~5x/day UTC to run
  `push_hybrid_if_missing.py` + `submit_best.py`.
- T4 Qwen route (14B/32B AWQ) is dead — 28-way eval concurrency starves
  inference, every request read-times-out -> 0.00. Do not resubmit T4.
- Hidden rerun keeps per-game cap 7920s (~110 games in 4 waves fit 9h).
- Leaderboard: 1st 18.81, top1% 5.28, top10% 3.51.

## Note

`run_local.py` resolves paths via `AGI3_HOME` / `KAGGLE_HOME` env vars
(default `~/kaggle/...`) — clone anywhere and set the vars, or mirror the
`~/kaggle` layout. `submit_best.py` / `push_hybrid_if_missing.py` resolve
everything relative to the repo checkout, so they work from any clone.
Cloud sessions suspend when idle: prefer one-shot `submit_best.py` runs
(or the scheduled Devin automation) over long-sleep watchers.

## Analysis notes (2026-09-21 evening UTC)

- Twin (NVFP4 baseline solver) per-turn stats: mean 595 tok/turn, p90 ~1.7k,
  2747/3020 turns <2k. Long-thinking pathology is a *tail* on flash-next
  (8 turns >16k, max 31k ≈ ~40min each at ~9 tok/s effective) — the
  MAX_OUTPUT=6144 cap's value there is bounding the worst case, not the median.
  On the 27B the pathology was the median (multi-k thinking every turn), so the
  `arc3-duck-qwen-27b-patched` candidate is the real cap test.
- Twin's vLLM watchdog: 0 restarts over the run — serving is stable.
- Ready-to-push candidate waiting for a GPU slot:
  `submit_ag3_duck_patched/` -> `takumuhata/arc3-duck-qwen-27b-patched`,
  dataset `takumuhata/taaf-27b-patched-bundle` (jakobbrggen bundle + patched
  tool_agent.py). `push_hybrid_if_missing.py` pushes it on the next free slot.
- anim zero-game autopsy (bp35/g50t/sc25): NOT timeouts (~1/game in ALL
  transcripts — noise). All games run a uniform ~52 turns in the 7920s cap;
  zero games simply never crack level-1 mechanics (e.g. bp35 step94/lvl1).
  Score scales with levels completed (ft09 5lv→47.62, lp85 6lv→41.67) —
  headroom is progress-per-turn, i.e. solver reasoning quality.
- Local-eval portability: the public-25 benchmark runs fully OFFLINE —
  `environment_files` + `arc_agi_3_wheels` ship in the competition dataset
  (download verified). Only blocker is model serving: flash-next is a
  125B-MoE (135GB NVFP4 → needs ~96GB-class GPU; impossible on Colab T4);
  27B FP8 (~30GB) needs ≥40GB VRAM. With such a GPU, the same harness can
  evaluate outside Kaggle (A/B-relative, not score-comparable).
