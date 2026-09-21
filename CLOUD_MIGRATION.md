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
# is unreliable. Prefer one-shot submission after the daily reset (~00:00 UTC):
cd arc-agi-3 && .venv/bin/python submit_now.py 15   # kernel_version arg

# local-machine alternative: sleep-until-slot watcher (fine on an always-on box)
nohup .venv/bin/python watch_and_submit.py >> watch_submit.log 2>&1 &

# poll a submission ref until it leaves PENDING
./poll_sub.sh <ref>   # run in bg
```

## Current state (2026-09-21, evening)

- AGI-3 today: OCEAN v14 submitted (ref `56408885`, PENDING — scored 0.14
  on the previous version)
- AGI-2 today: DSL v1 submitted (ref `56414919`; ~33.9 expected, top 0.5%)
- Best verified AGI-3 candidate: `arc3-duck-qwen3-8-27b`, public mean 4.79
- Queued for RTX6000: `arc3-duck-qwen3-8-flash-next-nvfp4-mtp`,
  `arc3-duck-flash-next-nvfp4-mtp-b`; `arc3-duck-anim-flashnext` pushes when
  a slot frees (`push_hybrid_when_free.py`)
- Score-aware watcher `arc-agi-3/watch_submit2.py`: wakes ~00:20 UTC,
  fetches each completed candidate's `summary.txt`, submits the highest
  verified mean (>= 4.0); Forge is late fallback. Uses `kernel_versions.json`.
- T4 Qwen route (14B/32B AWQ) is dead — 28-way eval concurrency starves
  inference, every request read-times-out -> 0.00. Do not resubmit T4.
- Hidden rerun keeps per-game cap 7920s (~110 games in 4 waves fit 9h).
- Leaderboard: 1st 18.81, top1% 5.28, top10% 3.51.

## Note

`run_local.py`, `watch_submit2.py`, `push_hybrid_when_free.py` resolve paths
via `AGI3_HOME` / `KAGGLE_HOME` env vars (default `~/kaggle/...`) — clone
anywhere and set the vars, or mirror the `~/kaggle` layout.
Cloud sessions suspend when idle: prefer one-shot `submit_now.py`-style runs
timed near the 00:20 UTC window over long-sleep watchers.
