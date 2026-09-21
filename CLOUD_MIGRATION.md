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

## Secrets

Org secrets already uploaded: `KAGGLE_ACCESS_TOKEN`, `KAGGLE_CREDENTIALS_JSON`.
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

# auto-submit at the next 00:10 UTC slot (edit kernel_version first)
nohup ../arc-agi-3/.venv/bin/python ../arc-agi-3/watch_and_submit.py \
    >> ../arc-agi-3/watch_submit.log 2>&1 &

# poll a submission ref until it leaves PENDING
../arc-prize-2026-agent-work/arc-agi-3/poll_sub.sh <ref>   # run in bg
```

## Current state (2026-09-21)

- OCEAN v13 submitted: ref `56408885`, public score **0.14**
- Forge submitted earlier: ref `56378196`, public score **0.08**
- Kernel v15 pushed (`takumuhata/arc-prize-2026-arc-agi-3-starter`);
  local watcher armed to submit it at the next 00:10 UTC window
- Paper-track writeup ready (1496 words); submission is browser-only —
  see `arc-prize-2026-paper-track/SUBMIT_INSTRUCTIONS.md`

## Note

`run_local.py`, `watch_and_submit.py`, etc. resolve paths relative to their
own location — clone anywhere and they work. `push_hybrid_when_free.py` is a
legacy script referencing dirs outside this repo.
