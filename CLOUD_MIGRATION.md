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

## Current state (2026-09-21, morning UTC)

- AGI-3 today: OCEAN v13 submitted 00:14 UTC (ref `56408885`, scored 0.14) —
  a stale `watch_and_submit.py`-style run burned the daily slot. Both old
  watcher scripts now delegate to `submit_best.py --watch`; kill any unsynced
  local copy so it cannot win the slot race tomorrow.
- AGI-2 today: DSL v1 submitted (ref `56414919`, PENDING; ~33.9 expected)
- Best verified AGI-3 candidates: `arc3-duck-qwen3-8-27b` public mean 4.79;
  `arc3-duck-qwen3-8-flash-next-nvfp4-mtp` public mean 4.63
- OceanCore local baseline (`run_local.py all 3000`, 25 dev games): mean 0.15%
  — matches the 0.14 LB score. Fallback-only; do not rely on it for placement.
- Queued for RTX6000 (batch GPU limit is 2, both taken):
  `arc3-duck-qwen3-8-flash-next-nvfp4-mtp` and `arc3-duck-anim-flashnext` (v1).
  The duplicate twin `arc3-duck-flash-next-nvfp4-mtp-b` was deleted to free the
  slot — the anim-hybrid (different solver variant) now evaluates instead of a
  redundant copy. Re-push it from a directory whose metadata carries that id if
  ever needed again.
- The two queued kernels are a real A/B, not duplicates:
  - twin uses keithtyser's baseline bundle: `tool_agent.py` unpatched,
    `LOCAL_ANALYZER_MAX_OUTPUT=0` (uncapped thinking+output)
  - anim-flashnext uses `takumuhata/taaf-anim-flashnext-bundle`: patched
    `tool_agent.py` — MAX_OUTPUT default 6144, a `finish_reason=="length"`
    nudge-to-emit retry, and progressive history shrinking after ≥2
    consecutive request failures. `bundle_hybrid/` and `taaf_src/` in this
    repo were synced to the deployed patched file (was previously stale).
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
- AGI-2: DSL symbolic-injection variant scored LB **29.72** (ref 56414919);
  plain `arc-agi2-lb33-89-perfpatch` claims ~33.89 — `AGI2_JOB` switched to
  plain. DSL prepends wrong symbolic attempt_1s and displaces model attempts.
- 27B patched-agent runs as v2 on the EXISTING `arc3-duck-qwen3-8-27b` slug
  (dataset `takumuhata/taaf-anim-27b-patched`); the separate
  `arc3-duck-qwen-27b-patched` kernel dir was removed — one GPU slot each,
  identical experiment.
- `submit_ag3_flashnext_patched` → `arc3-duck-flashnext-nvfp4-patched`:
  keithtyser nvfp4-mtp bundle + ported cap/salvage/shrink patches
  (dataset `takumuhata/taaf-flashnext-patched`, pinned `/2` since v1 was a
  stub while the 66MB tar finished processing). Pushes via pending_pushes.json.
- Kaggle CLI gotchas: `datasets create -r tar` silently fails on large bundles
  (stub create + `datasets version -r tar` works); `kernels_output` serves
  only the NEWEST version — an in-flight version hides the completed run's
  output; kernel-metadata `id` must equal the slugified title (409 else).
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
