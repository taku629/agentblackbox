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

## Colab verification environment (2026-09-22)

- `arc-agi-3/colab/q38_27b_benchmark.ipynb` — self-contained Colab notebook that
  reproduces the **27B public-25 benchmark** outside Kaggle (paid Colab, A100-40GB).
  It downloads the user's bundle + driessmit1's vLLM wheelhouse + the private FP8
  model + competition `environment_files`/`arc_agi_3_wheels`, recreates the
  `/kaggle` mount layout with symlinks, pre-installs vLLM into
  `<working>/vllm-site-packages` (wheelhouse -> PyPI fallback, arch-checked,
  stamp file makes setup_commands skip its own install), then runs the
  **unmodified** `takumuhata/arc3-duck-qwen3-8-27b` kernel via papermill.
- Env contract used: `TAAF_KAGGLE_WORKING_DIR=/content/working`,
  `TAAF_KAGGLE_BUNDLE_DIR`, `TAAF_KAGGLE_INPUT_PATHS` (maps all dataset/model
  refs to `/content` paths), `KAGGLE_GPU_TYPE=a100` (passes the substring GPU
  assert on A100-SXM4), `KAGGLE_GPU_COUNT=1`.
- Flash-Next NVFP4 kernels (anim etc.) are **not** Colab-runnable: ~135GB
  weights, no single-GPU fit. Only the 27B FP8 (~30GB) fits A100-40GB.
- **Cap-neutralization bug found + fixed**: every serving path wrote
  `LOCAL_ANALYZER_MAX_OUTPUT='0'` into `taaf_setup_env.json`, overriding the
  patched tool_agent default of 6144 — the output cap never actually engaged
  in ANY run so far (anim 8.21 came from retry-shrink/length-salvage alone).
  Patched to '6144' in `bundle_anim_v2/serving_setup.py`,
  `bundle_hybrid/serving_setup.py`, `flash_bundle/serving_setup.py`, and the
  inline PYSETUP of `setup_commands.json` (27B/taaf_src/taaf). Datasets
  re-uploaded: anim-flashnext-bundle v5, anim-flashnext-bundle-v2 v3,
  anim-27b-patched v3.
- Model download: `kaggle models instances versions download
  foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1 --untar` works
  with the user's credentials (private model, ~24GB, 16 shards).
- Expected runtime ~5-8h on A100; Colab 24h cap is fine. Score appears in
  `<working>/summary.txt` + `score.json` (same as Kaggle output).

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
- AGI-2: DSL symbolic-injection variant scored LB **29.72** (ref 56414919);
  plain `arc-agi2-lb33-89-perfpatch` claims ~33.89 — `AGI2_JOB` switched to
  plain. DSL prepends wrong symbolic attempt_1s and displaces model attempts.
- 27B patched-agent runs as v2 on the EXISTING `arc3-duck-qwen3-8-27b` slug
  (dataset `takumuhata/taaf-anim-27b-patched`); the separate
  `arc3-duck-qwen-27b-patched` kernel dir was removed — one GPU slot each,
  identical experiment.
- `submit_ag3_flashnext_patched` → `arc3-duck-flashnext-nvfp4-patched`:
  keithtyser nvfp4-mtp bundle, with cap/salvage/shrink patches applied
  IN-NOTEBOOK (copies the mounted bundle to WORKING_DIR, anchor-asserted
  source edits — no new dataset needed; `taaf-flashnext-patched` dataset
  exists only as a stub and is unused). Pushes via pending_pushes.json.
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
- Staged next experiments (auto-push on GPU quota reset ~Sun 00:00 UTC):
  `arc3-duck-qwen-27b-patched` (27B x anim x nudge/shrink — depth-vs-speed
  test), then `arc3-duck-anim-v2` (anim stack + Experiment 5 action-coverage
  hint: when stalled >=12 turns on a level the prompt surfaces per-action
  usage counts + MOUSE target coverage + untried valid actions — targets
  bp35-type stuck-loop failures). v2 lives in dataset
  `taaf-anim-flashnext-bundle-v2` + `bundle_anim_v2/`; production anim
  kernel stays pinned to v1.
- Verified GPU needed per lineage: flash-next is 125B-MoE (135GB NVFP4,
  needs ~96GB) — Kaggle-only; 27B FP8 ~30GB needs >=40GB (Colab Pro A100 or
  local A6000-class). User's 4070/5070 cannot serve either.

## Loop state (2026-09-23 UTC)

**Scores**: AGI-3 LB 3.98 (anim-flashnext v1; identical code scored 2.60 the
day before — hidden-set variance ~±0.7). AGI-2 LB 30.56 (plain perfpatch,
deterministic across 2 submissions; DSL variant 29.72 dropped).

**Deployed since last note** (all in dataset-latest, verified end-to-end by
download + ToolAgent instantiation):
- `tool_agent.py` budget clamp: `max_tokens = max(512, min(cap or 6144,
  remaining_s * 20))` — turns can't overrun remaining wallclock.
- Stagnation detector: ≥3 consecutive identical action batches with
  `board_changed=False` → prompt warning (self-clearing). Targets the
  bp35/sc25/g50t zero-level pattern of repeated no-op batches.
- Per-level action counter: `N actions used on this level` in the state
  line + prompt note that levels have hidden budgets (tn36's were 26-72).
- `prompts.py`: time-remaining awareness (below ~15% → commit best plan).
- `submit_best.py`: KNOWN_MEAN{anim-flashnext:8.21, qwen3-8-27b:4.97};
  27B pin removed (mapped−1 resolves newest completed run).

**Push queue** (auto-fires on GPU-quota reset — user reports Saturday per
Kaggle display; quota is 30h/week shared with RSNA comp):
1. `arc3-duck-anim-v2` — kept deliberately: its bundle is now byte-identical
   to bundle_hybrid (coverage hint ported everywhere), so it doubles as a
   same-code replicate for run-variance measurement.
2. `arc3-duck-anim-flashnext` v2 — all patches on the 8.21 stack.
3. `arc3-duck-flashnext-nvfp4-patched` — cap-only isolate (baseline).
4. `arc3-duck-qwen3-8-27b` v3 — 27B + anim solver + all patches (Colab-
   validated stack; expect ~5-6 on Kaggle's faster GPU).

**Dead ends confirmed**: AGI-2 DSL `dsl_all` + `s_search` beam search =
0/120 eval tasks (solves 11/150 train — eval distribution differs);
DSL contribution to LB ~0. AGI-2 stays on perfpatch (LLM+LoRA solver,
needs GPU — no local improvement path).

**Colab notes**: run2 on A100 = mean 1.81 (concurrency=6 + budget clamp);
Kaggle-vs-Colab gap is ~27 tok/s throughput (single A100-40GB + eager),
not solver logic. Runtime stopped to save units; future runs on
thatakumu@gmail.com (see ~/colab_notes.txt).

**Daily submissions**: automation `auto-526f75627a274282985a5a0d8744cf63`
ticks 00:25/08:25/16:25/22:05/23:35 UTC → push_hybrid_if_missing →
push_pending → submit_best (≥5.5 anytime, ≥4.0 after 22:00). T4 kernels
never submit.

## 2026-09-23 late — stagnation-trigger fix + real-LLM smoke path

- **Fix (16324ca, all 4 source trees + 3 datasets)**: the stagnation
  detector's `board_changed` trigger was dead on real games — the grid
  includes the HUD timer bar, so almost every action diffs ≥1 cell
  (bp35: 93/93 turns "changed"). Now solver.py emits `board_diff_cells`
  per action; the detector fires on `same batch ≥3 consecutive AND every
  diff ≤4 cells` (measured bimodal: HUD ticks ≤4, real moves ≥47).
  Real-data replay: bp35's MOUSE-batch repeat warns on turn 3; 13×RIGHT
  (47-cell) correctly stays silent.
- **CPU smoke-test path (reusable)**: `ollama pull qwen3:1.7b` (installed
  at 127.0.0.1:11434, OpenAI-compatible) → run the real harness on CPU:
  ```
  cd jak27/patched && PYTHONPATH=src/ARC3-Inference:src/tufa-arc-agi-framework/src \
  LOCAL_ANALYZER_BASE_URL=http://127.0.0.1:11434/v1 LOCAL_ANALYZER_MODEL_ID=qwen3:1.7b \
  LOCAL_ANALYZER_PROVIDER=vllm LOCAL_ANALYZER_ENABLE_THINKING=false \
  <repo>/arc-agi-3/.venv/bin/python -m inference.framework.run --game ft09 \
  --environments-dir <repo>/arc-agi-3/environment_files --max-actions 8 \
  --max-runtime-minutes 20 --concurrent-jobs 1 --experiment-dir /tmp/runX \
  --model local --analyzer-timeout 180
  ```
  Verified end-to-end: real prompt (with all new lines) → real model
  tool_calls → real python sandbox → real game env. 0.6B emits wrong
  markup; 1.7b emits valid `python` calls but weak args; 4b needs >120s.
- Grid-diff cell count is THE no-op signal (HUD bars tick every action
  at varying positions per game — no fixed mask works).

## 2026-09-24 — NoopGuard revival (same HUD root cause)

- **Known-Noop-Guard was dead on real games too**: `board_signature()`
  hashes the full grid incl. the ticking HUD strip → signatures never
  repeat → `is_known_noop` could never match; and `observe` used the
  contaminated `board_changed` → nothing ever recorded as a no-op.
- **Fix (tool_agent.py x4, deployed)**: track volatile cells (transitions
  with diff<=4 add their cells to `self._volatile_cells`, cap 64),
  `_masked_board_signature()` zeroes them, `real_change = len(diff)>4 or
  bool(diff - volatile)` feeds `NoopGuard.observe`. Functional test:
  noop recorded turn2+, blocked on retry; real moves unaffected.
- **`_describe_last_outcome` precision**: summary.max_board_diff_cells
  <=4 now tells the model "likely a no-op" instead of the old always-true
  "produced a board change" (it was misleading the model every turn).
- **Animation summary**: `summarize_animation(board_changed=diff>4)` —
  "final board identical, effect lives in intermediate frames" hint was
  dead (HUD tick made every board "changed"). Revived.
- **solver.py**: `board_diff_cells` now in events.jsonl records and
  batch-max aggregate; prompts.py documents the field for the model.
- **flashnext-patched stays cap-only isolate** (keithtyser datasets +
  notebook patcher, does NOT consume taaf bundles) — by design.
- **Kaggle OAuth**: access_token lives 12h; CLI "auto-refresh" uses a
  30-min post-expiry grace so calls fail ~30min with stale token. Force:
  ```
  .venv-kaggle/bin/python -c "
  from kagglesdk.kaggle_client import KaggleClient
  from kagglesdk.kaggle_creds import KaggleCredentials
  c = KaggleCredentials.load(client=KaggleClient()); c.refresh_access_token()"
  ```
- **Submission limits verified**: AGI-3 = 1/day (2nd submit 403s);
  leaderboard top ~19.4, we rank ~171/3274 at 3.98. AGI-2 = pinned v1
  30.56 deterministic; latest kernel output degenerate (harmless).
- **AGI-2 DSL dead deeper**: zero candidate fns on eval (not format) —
  no hybrid attempt_2 fill possible.

## 2026-09-24 — stub-LLM end-to-end verification (no GPU needed)

- `arc-agi-3/dev/stub_llm.py`: minimal OpenAI-compatible stub that plays a
  scripted action batch through the REAL `_handle_action` path. Run:
  `python3 stub_llm.py` (serves :18080/v1), then the same harness command
  as the ollama path with `LOCAL_ANALYZER_BASE_URL=http://127.0.0.1:18080/v1`
  `LOCAL_ANALYZER_MODEL_ID=stub`. ~15 real actions in ~1 min.
- Verified live on ft09 (MOUSE(3,3) x3 per turn, 5 turns):
  `board_diff_cells` in every event; stagnation warning fires turn 3
  ("same action sequence has repeated 3 turns"); action counter
  "N actions used on this level" correct; animation summary revived
  ("final board is identical ... effect only in intermediate frames");
  NoopGuard correctly does NOT record animated actions as no-ops.
- Local-only caveat: `inference.framework.run` default
  minimal_diagnostics=False crashes on mp4 render without imageio[ffmpeg]
  at run END (after games). Kaggle notebooks all pass
  minimal_diagnostics=True/run_as_submission — verified deployed kernels
  unaffected (movies/ + summary.txt present in arc3-duck-qwen3-8-27b
  output).
- Kaggle OAuth: forced refresh works; next access_token expiry 11:43 UTC
  2026-09-24. Ticks outside the 30-min stale-grace window self-heal.
- Dataset/git consistency re-verified: all 4 trees carry
  _volatile_cells/masked sigs/board_diff_cells/stagnation/coverage-hint.

## 2026-09-24 — real-data replay quantification of deployed patches

Replayed both winner runs' action events through the deployed NoopGuard +
stagnation logic (exact semantics: action_display signatures incl. MOUSE
coords; each action() call is a batch — verified batch_size=1 on all 3476
events; per-action grid diffs vs previous board with volatile masking):

- **NoopGuard**: would have blocked 11 actions (k27full run) + 2
  (af_out run) — concrete action-budget savings, zero false positives.
- **Stagnation detector**: would have injected 86 warnings (k27full:
  tu93=42, ls20=20, s5i5=6...) + 62 (af_out: bp35=12, s5i5=19, su15=15,
  sk48=9). Warnings are advisory text — whether the model heeds them is
  only measurable on the Saturday GPU runs.
- diff<=4 share: 285/1323 (21.5%) k27full, 446/2153 (20.7%) af_out —
  ~1 in 5 actions produces no real board change; that's the budget the
  new signals target.
- Replay tool: /home/ubuntu/replay_guard.py (uses deployed noop_guard.py).

## 2026-09-24 — turns are the binding constraint (new prompt deployed)

- Winner-run forensics: EVERY game ended NOT_FINISHED on wall-clock, and
  analysis-turn count is ~51-57 per game REGARDLESS of actions taken
  (wa30: 237 actions/51 turns = 4.6/turn; sb26: 30/55 = 0.55). Turn count
  is fixed by model latency (~2.5 min/turn x ~52 ~= 130 min game budget)
  -> per-turn action rate is what separates good games from dead ones.
- prompts.py (x4, uploaded): new line tells the model a game budget is
  only ~40-60 turns and inspection-only turns spend the same budget --
  prefer >=1 action/turn and multi-action() calls when steps are
  predictable. (batch_size=1 on all 3476 real events: the model never
  batched before.)
- tool_agent.py (x4, uploaded): stagnation warning now appends
  "Valid actions outside the repeated sequence: ..." (functional tested:
  lists untried names, silent when all tried, no false fire).
- coverage-hint verified live: fires at turn 12 with per-level action
  usage counts + MOUSE target count + untried list (unit replay).

## 2026-09-24 Research findings + P3.1 adoption (commit 20789d9)

- **Programmatic solvers already tried, failed**: repo's own Forge Pathfinder
  (BFS+RL, zorojuro lineage) submitted 9/20 -> LB 0.08; OCEAN induction -> 0.14.
  Both 25-50x below LLM agent. Public BFS recipes (zorojuro v9 etc.) top ~0.39
  LB. Do NOT reinvest in search-on-source direction.
- **Tufa Labs confirmed**: all top public notebooks build on their Duck/TAAF
  harness (ours). They will NOT open-source at milestone 2 (fear of 4000-copy
  noise). Scott Le Grand reached #38 (5.19) "on harness fixes alone" ->
  harness-level work like ours is the validated path at our scale.
- **True scoring rule (reverse-engineered by gedouluhui, verified vs telemetry)**:
  level score = min(115, (baseline_actions/your_actions)^2 * 100); every action
  incl. in-level RESET costs +1; game score = level-index-weighted mean
  (later levels worth more); LB = unweighted game mean. Competition mode is
  single-play -> per-game luck can't be resampled; opening-minutes hypothesis
  quality is the dominant variable (bimodal outcomes).
- **Adopted (commit 20789d9, all 3 datasets re-uploaded)**: perception_hints.py
  (~120 LOC, CC0 from gedouluhui/taaf-p3-bundle) - deterministic connected-
  component click candidates injected for MOUSE games; quadratic scoring
  prompt line. Validated on real bp35 board (188 objects, HUD strip filtered).
- **Their negative results (pre-registered, worth respecting)**: 64k context ->
  throughput collapse; temp 0.6->0.3 -> exploration collapse; archetype
  playbook -> premature commitment. Keep context 32k, temp 0.6.
- **Community local->LB band**: local 8-16 -> LB 4-7 (ours 8.21 -> 3.98 fits).
  Run-to-run sigma ~0.8-0.9 on 25 games; our 2.60/3.98/3.13 LB spread on same
  code is consistent. Final ranking uses best-2 submissions -> keep daily runs.
- **Future lever flagged by two sources**: behavioral cloning / LoRA on winning
  transcripts (Scott Le Grand 11.04->12.97; P3.1 plans same). Needs GPU.

## 2026-09-24 late: ACTION7 fix + forum intel

- **ACTION7 (UNDO) was silently unusable**: action_names.py lacked the mapping
  (forum: Duck-lineage agents hit index errors on ACTION7). Fixed + datasets
  re-uploaded. If a hidden game offers ACTION7, v3+ kernels can now use it.
- DeepSeek V4 Flash on RTX-6000-Pro: multiple teams report not viable
  (quant/prune hacks, slow decode, vision broken). Not worth our effort.
- CoTRD decoding paper exists (no scores published) - low priority.

## 2026-09-24 research round 2: hidden-set structure + failure catalogs

- **Hidden eval is engine-only**: no game .py files on the box for hidden games
  (confirmed in forum + explains Forge 0.08: find_game_source never finds them).
  Search-on-source is IMPOSSIBLE at LB time -- direction permanently closed.
- **Hidden set ~110 games**, ~9h, 4 waves, single-play. Two submissions count
  for final ranking (keep daily submissions; hedge variance).
- **STaR LoRA (manas joshi, CC0)**: winning-trajectory SFT on duck harness
  gave 1.25 -> 1.94 LB (rank ~300 -> ~133). Public datasets:
  arc3-sft-trajectories (chat-JSONL), arc3-duck-lora-sft (Qwen3.6 adapter),
  duck-eval-results. Gotchas: assistant-tokens-only, flatten OpenAI tool-calls,
  keep Qwen3_5ForConditionalGeneration multimodal merge, copy processor
  configs. Needs GPU -- candidate for post-Saturday exploration.
- **Fususu failure catalog (matches P3.1)**: multi-LLM roles 5x calls too slow;
  strategy library -> premature-classification bias; concurrency >16 saturates;
  Gemma-4-31B better vision but worse logic; manually-played training data not
  better than self-play traces; thinking-off fast but careless.
- **MULTIMODAL_UPSCALE=4 confirmed correct** (writeup tuned it; forks running
  8 have no posted A/B). Our config already 4. No change.
- Unofficial game names (Tong Hui Kang): bp35 buoyancy-puzzle, cn04 connector-
  network, tu93 traverse-unharmed, tn36 toggle-navigation, sb26 sequence-
  builder, wa30 warehouse-agents, ls20 lock-smith, vc33 volume-control,
  lp85 loop-placement, sk48 sliding-kebab, ar25 axis-reflections...
  (full list in discussion 738294)

## 2026-09-24 late2: own-run SFT extraction (LoRA prep, CPU-only)

- Our winner-run artifacts contain FULL per-turn transcripts (system/user/
  assistant+tool_call/tool-result). dev/extract_sft.py converts them into the
  public-dataset chat-JSONL format. Verified: 0 ordering issues.
- Yield: 17 winning games / 2398 msgs (27B run k27full) + 22 winning games /
  3587 msgs (flash-next af_out, mean 8.21) — vs public dataset's 22 convs from
  a ~1.25-mean model. Ours comes from a 4-6x stronger agent.
- Files: /home/ubuntu/research/sft/sft_ours_{27b,flash}.jsonl (on box).
  Train assistant-tokens-only when GPU returns; merge as
  Qwen3_5ForConditionalGeneration (multimodal) + copy processor configs.

## 2026-09-24 late3: ACTION7 fix is score-enabling (VERIFIED on own transcripts)

- ACTION7 is a PER-GAME mechanic slot, not necessarily undo: in ar25 it is the
  rotation action the model needs ("ACTION7 rotates it, and the goal is to
  align the piece with the target"). Engine offers ACTION7 in valid_actions on
  at least 6 public games (ar25, bp35, lf52, sb26, sk48, su15).
- PRE-FIX BUG (verified): to_engine_action('ACTION7')->None made the whole
  action batch fail normalization ("Unknown action at index N", executed:False,
  no budget spent). Our transcripts show the model correctly diagnosed ACTION7
  as the mechanic and tried to call it — ar25 mentions it 73x, bp35 370x,
  su15 244x, sk48 167x across the two winner runs. Every call was rejected.
  Keeping the map fix is likely a direct score unlock on those games.
- Tufa deliberately hid UNDO ("model undoes big batches, wastes energy") but
  that reasoning applies only where ACTION7==undo semantics; here it's a real
  mechanic, so mapped is correct (P3.1 shipped the same fix). UNDO alias kept
  as a safety net for the model writing "UNDO".
- Tufa baseline: Duck + Qwen3.6-27B public mean 1.60 ± 0.45 (20 tries/game).
  Ours: flash-next 8.21 local — ~5x their published baseline.
- Their known gap (model sees no animation feedback; sb26/tn36 suffer) is
  already covered by our animation-summary hint — we're ahead on that axis.

## Baseline level table (for A/B vs Saturday run)

Per-game max level reached (27B | flash-next):
ar25 2|2, bp35 2|1, cd82 2|2, cn04 2|2, dc22 2|3, ft09 4|5, g50t 1|1,
ka59 1|2, lf52 1|2, lp85 2|6, ls20 2|2, m0r0 1|2, r11l 2|2, re86 3|3,
s5i5 2|2, sb26 5|2, sc25 3|1, sk48 1|2, sp80 1|3, su15 2|2, tn36 1|2,
tr87 1|2, tu93 3|4, vc33 3|4, wa30 2|2. EVERY game ends NOT_FINISHED
(timeout) — turns are the binding constraint, so early-level speed is the
score lever, not deep progress. Watch ar25/bp35/su15/sk48/lf52/sb26 for
ACTION7-fix uplift; tu93/ls20 for stagnation-warning effect.

## 2026-09-25: STaR LoRA pipeline — BUILT + CPU-verified (bold pivot)

Bet: improve the MODEL, not just the harness. Public pilot (manas joshi,
CC0): LoRA r16/a32/dropout .05 on Qwen3.6-27B bf16 from 9 convs / 9116 label
tokens → LB 1.25→1.94 (+55%). Our distilled set is ~470x larger.

### Assets (all live)
- packed_sft.jsonl: 3823 samples, 0 skipped, avg 7277 tok / 1949 label tok,
  total 7.45M label tokens. Sources: FOUR own runs (af_out 22g/3588msg,
  k27full 17g/2398msg, k27v2 17g/2250msg, nvfp4_out 16g/2604msg — distinct
  trajectories of the same games = more diverse action patterns)
  + public STaR convs. ~800x the public pilot's 9116 label tokens. Validated: every sample has ≥1 user msg, ends on the
  assistant target turn (98% with tool_calls).
- Dataset takumuhata/taaf-duck-sft-v1 (packed_sft + train/pack/extract + tok38).
- Dataset takumuhata/taaf-anim-27b-lora = anim-27b-patched bundle + vLLM
  --enable-lora wiring (base served as Qwen/Qwen3.8-27B-bf16, adapter alias
  duck-27b-lora → solver requests route to adapter via SERVED_MODEL_NAME).
- Kernel dir sft_train/ → arc3-duck-sft-train: bf16 base (rahim3 dataset) +
  packed_sft → PEFT adapter out to /kaggle/working/lora_adapter.
- Kernel dir submit_ag3_lora/ → arc3-duck-anim-27b-lora: 27B eval kernel
  serving bf16 base + adapter (datasets: taaf-anim-27b-lora, wheelhouse,
  rahim3 bf16, taaf-duck-lora-v1). NO model_sources.

### Template/label specifics (verified on real samples)
- Qwen3.8 chat template: expects tool_call.arguments as DICT (normalize
  OpenAI JSON-string args before templating — the public 'flatten' gotcha).
- Template refuses prefixes with no user query; head context is cut at the
  first '<|im_start|>user' boundary. Prefix-stable across turns.
- Labels only on assistant segments minus '<|im_start|>assistant\n' header
  (matches 'assistant-tokens-only' guidance from the pilot).
- Template injects a 'Reasoning effort is set to xhigh...' preamble into the
  system block — identical at train and serve time (vLLM uses same template).

### Saturday sequence (after 01:00Z reminder → re-enable automation)
1. push_pending pops arc3-duck-sft-train first → ~4-6h train on RTX Pro 6000.
2. `kaggle kernels output takumuhata/arc3-duck-sft-train` → adapter files →
   `kaggle datasets create -p <dir>` as takumuhata/taaf-duck-lora-v1.
3. Manual push submit_ag3_lora/ (NOT in pending queue — would fail-fast on
   missing adapter). Eval → summary.txt mean vs 8.21 baseline.
4. If adapter regresses: no submit (best-2 keeps 3.98); if it wins: submit.
- One-shot helper: `arc-agi-3/sat_lora_pipeline.sh` runs steps 2+3 above
  (adapter fetch → dataset create → lora kernel push) once the train kernel
  finishes. Run it manually after checking train output quality.
- Note: packed_sft samples are text-only while prod turns include the grid
  image — same text-only regime as the published pilot (worked: +55%).
  Adapter teaches action style/tool format on language layers; the vision
  tower stays frozen.
- rahim3/qwen3-8-27b-bf16 has tokenizer+vocab+merges+preprocessor configs
  (multimodal) but no separate chat_template.jinja — vLLM uses the embedded
  template in tokenizer_config.json (same as serve-time behavior).
- vLLM cmd now pins --max-lora-rank 16 (adapter is r=16, default ceiling).

## 2026-09-25b: AGI-2 eval 0/120 resolved (non-production)

eval_local.py's 0/120 on the eval set was NOT a production bug: it measures
the dev-side dsl_all.py battery (38 fixed-pattern solvers). All 38 return
None on eval tasks (no pattern fits) vs 11/150 on train — the eval set is
designed so memorized transforms fail; a fixed battery scoring ~0 there is
expected. Production AGI-2 kernel (arc-agi2-lb33-perfpatch, LB 30.56) is an
AIMO-style stack instead: sorokin/qwen3_4b_grids15_sft139 (4B Qwen3 SFT) +
unsloth LoRA + turbo_dfs beam search + train-time aug (n=16) + eval-time aug
(n=2) + NLL re-rank — unrelated to the DSL battery. AGI-2 improvements are
GPU-gated (aug/DFS budget tuning needs quota); deprioritized behind the
AGI-3 LoRA bet.

## 2026-09-25c: PAUSED (user needs GPU quota for another job)

- Main automation auto-526f75627a274282985a5a0d8744cf63 DISABLED — no
  submits/pushes until the user asks to resume (was first paused for Sep 25,
  then the Sep-26 09:00JST revival reminder was deleted at user request
  because the Sep-26 morning GPU quota is reserved for their other job).
- resume = re-enable automation + run push_pending (order unchanged:
  sft-train first), then sat_lora_pipeline.sh once the adapter is out.

### Training-time budget fix (train_lora.py + sft_train nb updated, dataset v4)
- Measured real tokens: packed_sft = ~27.9M input tok/epoch (mean 7294,
  p90 hits 8192 cap), ~7.25M label tok. At ~200-300 tok/s on RTX PRO 6000
  that is ~25-40h/epoch — blows the 30h/wk quota AND the kernel cap.
- Kernel now runs EPOCHS=1 + SUBSAMPLE=1200 (~8.7M tok, ~8-12h est).
  train_lora.py takes SUBSAMPLE env (deterministic seed-1234 pick of packed
  lines, sub-sampled before encoding to save the O(n^2) render cost too).
- Bug fixes in train_lora.py: grad-ckpt now owned by Trainer
  (gradient_checkpointing=True + use_reentrant=False) and peft wrap happens
  first — the old order never enabled input grads (element-0-no-grad crash).
  enable_input_require_grads called explicitly; use_cache=False applied to
  both config and config.text_config; save_strategy=steps every ~quarter
  epoch (keep 2) so a timeout still leaves a usable adapter checkpoint;
  warns if any LoRA target lands on vision modules; remove_unused_columns
  off (PEFT forward signature introspection can drop needed cols).

### Resume-time quota math (30h/wk budget)
- sft-train ~8-12h (SUBSAMPLE=1200) -> lora eval ~5-8h = ~13-20h for the
  LoRA bet. Remaining ~10-17h fits ~2 of the 3 queued eval kernels
  (anim-flashnext v2, flashnext-patched, 27B-patched) — if quota runs short,
  drop 27B-patched first (the lora eval already covers a 27B run).
- sat_lora_pipeline.sh hardened: falls back to newest checkpoint-N adapter
  if the train kernel times out (mid-run ckpts every ~quarter epoch), and
  auto-switches datasets create->version on reruns.
- Verified end-to-end on CPU: SUBSAMPLE env (40-line pick -> 0.3M tok),
  kernel metadata has all 4 dataset_sources incl. taaf-duck-lora-v1,
  vLLM wiring: --enable-lora --max-lora-rank 16 --lora-modules
  duck-27b-lora=<LORA_PATH>, LOCAL_ANALYZER_MODEL_ID=duck-27b-lora.

### vLLM-LoRA serving risk assessment (checked v0.19.0 source)
- CONFIRMED: Qwen3_5ForConditionalGeneration is in the 0.19 model registry
  (qwen3_5.py), inherits Qwen3VLForConditionalGeneration which implements
  SupportsLoRA, and uses _mark_language_model so LoRA applies to the LM
  tower only — exactly where our targets land (vision modules use fused
  qkv names and are untouched).
- Known-era bug (vllm#28640, v0.11/0.12): lora_shrink assert when adapter
  contains vision params or DS-Z3/FSDP-sharded 1-D tensors — neither
  applies to our PEFT-Trainer export; also reportedly fixed in later
  releases. Residual risk remains (first real check is eval-kernel boot).
- Qwen3_5 is IsHybrid (linear-attn + full-attn mix): our targets only hit
  full-attn + MLP projections — partial coverage is expected and fine.
- FALLBACK if --enable-lora fails at serve time: merge adapter into base
  inside the eval kernel before vllm starts (transformers+peft already in
  wheelhouse): load base bf16 + adapter, merge_and_unload, save to
  /kaggle/working/merged27b (~54GB — check scratch disk; else merge
  shard-by-shard streaming), serve that dir with the same cmd minus
  --enable-lora/--lora-modules and SERVED_MODEL_NAME unchanged.
  train_lora.py already supports SAVE_MERGED_DIR for an in-kernel merge,
  but the merged 54GB exceeds the kernels-output download path — prefer
  merge-inside-eval-kernel.

### Pilot-adapter forensics (justforgags/arc3-duck-lora-sft, inspected)
- Spec identical to ours (r16/a32/7 targets/3ep/1e-4); trained on
  vrfai/Qwen3.6-27B-FP8 — layer geometry MATCHES our bf16 27B exactly
  (64 layers, h5120, mlp 17408; naming difference is community naming,
  same qwen3_5 arch). train_loss 0.36 on 9 convs / 9116 label tokens.
- KEY MISMATCH caveat: pilot adapter keys are `base_model.model.model.
  layers.N.*` (text-only repack); ours will be `base_model.model.
  language_model.layers.N.*` (VL checkpoint). vLLM 0.19 strips
  'base_model.model.' and matches against its own module tree —
  our naming is the expected VL path (same layout as the Qwen3-VL-8B
  case confirmed working in vllm#31278). The PILOT adapter as a fallback
  would need a key rename (insert 'language_model.') before use.
- Tokenizer note: pilot ships chat_template.jinja separately; our base
  embeds the template in tokenizer_config.json — vLLM reads the served
  (base) model's tokenizer either way; adapter tokenizer files inert.
- Stratified SUBSAMPLE now in train_lora.py (floor N/ngames + proportional
  remainder, seed 1234) — 1200-sample pick gives every game >=13 samples
  (was 22:1 skew su15:279 vs k008:13). Dataset taaf-duck-sft-v1 v5.
- Official tech report confirms 5x human-baseline action cap per level —
  aligns with the (baseline/yours)^2 prompt already deployed.
- Pilot adapter structure (deeper): self_attn q/k/v/o on only 16 of 64
  layers + mlp gate/up/down on all 64 — Qwen3.5 is hybrid; linear-attn
  layers have no q/k/v/o names, so our identical target list lands the
  same way automatically. 256 lora pairs, dims verified compatible.
- Plan-B merge script: arc-agi-3/dev/merge_lora_stream.py — streams base
  shards + adapter, suffix-matches on `layers.N.<mod>.weight` (tolerates
  both VL `model.language_model.layers` and text `model.layers` nesting),
  verified key-parsing against the real pilot adapter (256/256 resolve).

## 2026-09-26: Colab QLoRA path (user has Google AI Pro / Gemini Colab access)

- `sft_train/colab_qlora_train.ipynb` — self-contained Colab notebook:
  creds upload -> kaggle datasets download (base 44GB + sft) -> QLORA=1
  train -> adapter upload to takumuhata/taaf-duck-lora-v1. Needs an
  A100-40GB runtime (~3-4h for SUBSAMPLE=1200; L4-24GB likely OOMs at
  seq 8192 from the 248k-vocab logits — MAX_LEN=4096 is the fallback).
- train_lora.py gained QLORA env: BitsAndBytesConfig nf4 + double-quant +
  bf16 compute, peft.prepare_model_for_kbit_training. bf16 path unchanged
  (Kaggle RTX6000 remains the reference run if quota frees first).
- This DECOUPLES the scarce resource: Colab trains the adapter while
  Kaggle GPU stays free for the user's other job; only the ~7h eval
  kernel still needs Kaggle quota.

## 2026-09-29: QLoRA training DONE on Colab A100-80GB (adapter published)
- Drove the run in Colab (thatakumu@gmail.com, authuser=1) on
  notebook 1iP5fqBc_dodJ-rRegb6MnMEWWJuQf3iN, nohup python3
  /content/sft_data/train_lora.py: SUBSAMPLE=1200 stratified, 1 epoch,
  150 steps, QLORA=1 4-bit NF4, MAX_LEN=8192, r16/a32 LoRA on
  q,k,v,o,gate,up,down (7 module kinds, 256 lora_A/B pairs, 79.7M params).
- Result: train_loss 0.5925 avg, last logged 0.5564->0.5666 (from 0.7127
  at step 10; steady descent, plateau ~0.58 mid-run). grad_norm stable
  0.2-0.34. Runtime 35090s ~9.75h (~233s/step: hybrid delta-net layers
  take unfused PyTorch path without flash-linear-attention).
- Checkpoints ck37+ck111 backed up locally (/home/ubuntu/lora_ckpts/).
- Published: takumuhata/taaf-duck-lora-v1 (private dataset) =
  adapter_model.safetensors 304MB + adapter_config.json + chat_template.jinja
  + tokenizer.json + tokenizer_config.json. Note adapter_config
  base_model_name_or_path=/content/base_27b (harmless — eval kernel loads
  bf16 base itself, peft only needs the tensors).
- Colab pitfalls hit (see /home/ubuntu/colab_notes.txt): kaggle CLI buffers
  whole downloads in RAM (stream via requests), KAGGLE_KEY secret is a
  FOREIGN account token (403 on private datasets — upload via Files panel
  + push datasets locally), terminal typing drops chars (use .sh uploads),
  qwen3_5 needs transformers>=5.x + peft>=0.21, transformers 5.x dropped
  TrainingArguments.warmup_ratio (script filters kwargs by signature).
- NEXT (needs Kaggle GPU — paused per user): push submit_ag3_lora/
  (arc3-duck-anim-27b-lora) as before: base rahim3 bf16 + this adapter via
  --enable-lora, ~7h eval. If verified mean >8.21, submit.
- Alternative future eval path: run the same eval on the Colab A100 (saves
  Kaggle quota) — needs the eval notebook + datasets ported; the harness
  itself is engine-only so it should work, unverified.

## 2026-09-28 (later): LoRA eval ported to Colab (Kaggle quota stays free)

- New: `submit_ag3_lora/colab_lora_eval.ipynb` + `colab_eval_setup.py` +
  `make_eval_nb.py` (regenerate the notebook after editing the setup script).
- New private dataset `takumuhata/taaf-colab-deps-v1` = arcengine/arc_agi site-packages
  + `environment_files/` (25 public game env dirs, nested zips inside the dataset zip).
- Colab notebook flow: pip vllm==0.19.0 (PyPI, not the H100 wheelhouse) →
  KAGGLE_CREDENTIALS secret → stream-download bundle/adapter/model/deps →
  vLLM `--enable-lora` server (same flags as Kaggle kernel) → patch
  `arcade_spec.environments_dir` → `bm.run()` 25 games ×1 pass → mean from
  `working/benchmark.json` game_runs final_score, saved to
  `/content/lora_eval_result.json`.
- Auth: datasets downloaded via `Authorization: Bearer <access_token>` streaming
  (credentials.json is OAuth-style, NOT basic-auth key). On 401 the notebook runs
  `kaggle datasets list -m` which auto-refreshes the token on disk, then retries.
- Needs A100 80GB (54GB weights). Verified mean >8.21 ⇒ submission candidate —
  but submitting still needs Kaggle GPU + user resume.

## 2026-09-29: Colab eval RUNNING + real Kaggle download API flow documented

- Eval is LIVE on the Colab A100-80GB runtime (colab_lora_eval.ipynb): all 4
  datasets downloaded (44GB base, 18 shards), vLLM serving base+LoRA under the
  `duck-27b-lora` alias, `bm.run()` 25 games x 1 pass underway (~6-8h).
- REAL Kaggle dataset-download flow (kaggle CLI broken on the runtime):
  credentials.json holds OAuth {refresh_token, access_token(expires ~24h),
  username}. Mint a fresh token via
  POST https://www.kaggle.com/api/v1/access-tokens/generate
    json={'refreshToken': rt, 'apiVersion': 'API_VERSION_V1'}
    headers={'Authorization': 'Bearer ' + old_access_token}
  -> {'token': new}. Then POST
  https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset
    json={'ownerSlug': o, 'datasetSlug': s} + Bearer -> 302 signed GCS URL.
  Works for PRIVATE datasets. Legacy GET /api/v1/datasets/download/<slug>
  only works for PUBLIC. HTTP Basic (username, KAGGLE_KEY) only works with
  takumuhata's OWN api key — org-secret KAGGLE_KEY is a different account (403).
  Working downloader template: /tmp/colab_dl2.py (also hosted on paste.rs).
- deps dataset (taaf-colab-deps-v1) contains PLAIN dirs arc_pkgs/ +
  environment_files/ — NOT nested zips. make_eval_nb.py cell 4 now searches
  both /content/deps_unz and /content/deps.
- bm.games[0].arcade_spec is a FROZEN attrs instance: assign via
  object.__setattr__(spec, 'environments_dir', Path(ENV_DIR)).
- Colab GUI pitfalls: Files-panel uploads land in the PANEL's browsed dir
  (were at /, not /content); notebook TERMINAL typing silently drops '_', '*',
  '"' (a 'deps_unz' mkdir became literal 'deps?unz' — glob quirk) — type
  commands in a CODE CELL or upload a .sh script instead; Ctrl+F10 doesn't
  reach Colab, use the cell play buttons / Ctrl+Enter inside a focused cell.
- make_eval_nb.py updated with all of the above; bfdcbdb..616241a pushed.

## 2026-09-29 (later): Colab LoRA eval COMPLETE — mean 1.669, NOT a candidate

- colab_lora_eval.ipynb ran clean on A100-80GB: 25 public games x 1 pass in
  2h12m. mean final_score = 1.669 (median 0.0, all 25 runs ended 'gave_up').
- Scores >0 (9 games): ft09 14.29 (lv2/6), vc33 10.71 (lv2/7), cn04 4.76,
  lp85/s5i5/sb26 2.78, re86 2.09, su15 1.28, bp35 0.27. 16 games 0.0.
- vs the 8.21 submission baseline: ~5x lower. NOT a submission candidate.
- Likely causes: (a) QLoRA adapter was trained against NF4-quantized base but
  served on bf16 weights — the adapter absorbs quantization correction, not
  pure behavior; (b) 1 epoch / 1200 stratified samples may be underfit;
  (c) agent gives up early (many runs <50 actions).
- bf16 LoRA retrain on Kaggle (original plan, trainer already supports it)
  is the clean retry when GPU quota frees. Result saved:
  /content/lora_eval_result.json on the runtime; transcripts in
  /content/working/transcripts/.

## 2026-09-30: LoRA eval root-cause narrowed — adapter degraded the model (env verified identical)

- Verified deps dataset env files carry the SAME instance IDs as the official
  eval (ft09-0d8bbf25, vc33-5430563c, ...) — apples-to-apples confirmed.
- Un-adapted 27B on identical 25g eval (k27v2) = **4.97**; anim-flash = 1.81;
  base+QLoRA-adapter = **1.669**. If the adapter had failed to bind, we'd
  reproduce ~4.97 — instead it actively cut score 3x. Per-game: ft09 14.29
  (vs base 47.6), lp85 2.78 (vs 16.7), ar25 0 (vs 8.3); wins only vc33/cn04.
- Adapter forensics: deltas ~2-3x pilot norm (med 0.3 vs 0.1) — consistent
  with absorbing NF4 quant corrections (QLoRA mismatch theory).
- Leading explanation: NF4-trained deltas applied to bf16 weights over-
  correct. Underfit (1ep/1200 samples, loss plateau ~0.58) is the fallback.
- NEXT experiment prepared: `submit_ag3_lora/colab_merged_eval.ipynb`
  (make_merged_eval_nb.py + merge_setup.py + MERGED_MODEL_PATH mode in
  colab_eval_setup.py). Proper QLoRA merge = dequantize NF4 base + add deltas
  -> /content/merged_27b bf16 -> serve under duck-27b-lora. 7 signal games
  (subset of both prior evals), ~2h total incl. 44GB re-download.
  Verdict: merged-mean recovers toward ~5 -> adapter fine, ship merged model
  (Kaggle, no --enable-lora); stays ~1.7 -> adapter bad, retrain bf16.
  Runtime currently dead — needs a fresh Colab A100-80GB when user resumes.

## 2026-09-30 (later): merged-eval experiment — VERDICT: adapter itself is bad

- Ran colab_merged_eval.ipynb on Colab A100-80GB. Proper QLoRA merge
  (dequantize NF4 base + add deltas) produced a MIXED-PRECISION dir:
  only the 256 LoRA-targeted modules dequantized to bf16 (570 tensors);
  1818 untargeted modules stayed nf4 (nested_absmax/nested_quant_map U8
  keys) — vLLM cannot serve that. 17.9GB single safetensors at
  /content/merged_27b (kept on runtime; delete to reclaim disk).
- PIVOTED to the cleaner test: vLLM native bnb serving =
  --load-format bitsandbytes --quantization bitsandbytes + --enable-lora
  (nf4 base + adapter = the EXACT training-time configuration; merge not
  needed). Confirmed working: weights loaded in 18.4GiB (nf4 size) with
  LoRA active. Injected via kernel env:
    os.environ.pop('MERGED_MODEL_PATH')
    os.environ['VLLM_EXTRA_ARGS'] = '--load-format bitsandbytes --quantization bitsandbytes'
  (VLLM_EXTRA_ARGS is appended to the server cmd inside
  start_vllm_server; cell 7 rewrites colab_eval_setup.py from SETUP_SRC
  every run — patch the kernel env, not the file.)
- RESULT (7-game subset, identical env instance IDs):
    merged nf4+LoRA (faithful)   mean 1.76
    bf16+LoRA    (mismatched)    mean 4.95 on same subset (1.669 all-25)
    base 27B     (un-adapted)    mean 13.87 on same subset (4.97 all-25)
  Per-game nf4+lora: re86 8.33, lp85 2.78, vc33 1.22, rest 0.00.
- VERDICT: **adapter bad** — serving precision was NOT the cause; the
  trained deltas actively degrade the model under any precision.
  Quantization-mismatch theory dead. ~10h QLoRA run produced a
  destructive adapter (train_loss ~0.58 but eval collapsed).
- Runtime notes: cell editor drops newlines (xdotool typing) — write
  cells as single-line semicolon statements or upload .py scripts via
  the Files chooser; Terminal also drops chars, use xdotool type+Return.
  transformers 5.x was needed for the merge attempt but vllm 0.19
  requires <5 — downgraded to 4.57.6; bitsandbytes 0.50.2 installed.
- IN FLIGHT: over-correction probe — scaled adapter lora_B x0.4 ->
  /content/lora_adapter_s040 (scale_adapter.py), kernel env repointed,
  server re-booting for a second 7-game eval (~1.5h). If it recovers
  meaningfully: deltas too strong -> retrain lower LR/alpha or eval an
  early checkpoint (ck37/ck111 live in /home/ubuntu/lora_ckpts/).
  If it stays ~1.7-2: retrain bf16 LoRA (A100-80GB can hold bf16 54GB
  + adapter optim states) or revisit SFT data quality.

## 2026-09-30 (final): scaled-adapter probe + free forensics — LoRA route parked, needs data rethink

- Scaled-adapter (lora_B x0.4) 7-game eval: partial mean ~2.34 when the
  Colab runtime was RECYCLED mid-run (compute-unit exhaustion). Scores:
  re86 5.56, vc33/cn04/ar25 ~1.4-2.4, ft09/sc25 0.0. Better than faithful
  1.76 but nowhere near bf16+adapter 4.95, let alone base 13.87 —
  over-correction rescue FAILED.
- Delta-norm progression (||B||x||A|| med): ck37 0.81 -> ck111 1.00 ->
  final 1.06. Adapter was born ~3x pilot magnitude and stayed there —
  not a late-training blowup. ck37 eval would likely also be bad.
- SFT data forensics (packed_sft.jsonl): 58.6% of assistant msgs have
  EMPTY content (bare tool_call); 48.3% of samples END on a bare tool
  call; median assistant len 0. Most supervised tokens = action-format
  JSON, not reasoning -> plausibly taught 'act without thinking'.
  Give-up-language hypothesis dead (0.3%). Masking + chat template
  verified correct and identical between train tok and adapter jinja.
- VERDICT (final): adapter destructive under any serving precision;
  scaling/mismatch hypotheses exhausted. The recipe (short windows,
  reasoning-thin, 1ep on 1200 samples, deltas 3x pilot) needs a DATA
  rethink — full-length convs, reasoning-dense samples — not a
  precision/config tweak. Each Colab retrain ~10h A100.
- Colab compute: user account drained to ~8.5 units (~1h A100) — runtime
  recycled all /content (44GB base gone); disconnected to preserve
  balance. GPU LoRA path CLOSED until units recharged or Kaggle quota.
- Next lever remains the harness route (proven): 8.21-run kernels +
  deployed perception/scoring-prompt patches; resume Kaggle automation
  when user frees quota. bf16+adapter 4.95-subset vs base 13.87-subset
  means even the 'working' adapter config costs 3x — LoRA wins nothing
  short of a recipe overhaul.

## 2026-10-01 — Stale-master automation regression (ROOT CAUSE of bad submissions)

**Finding**: the daily-submit automation (`auto-a05353b07e244f6f891382fc34bb7b9f`,
ticks 04:45/10:45/16:45/22:45/23:45 UTC) clones the repo's DEFAULT branch each
run. origin/master = 909ac93 (mid-history of the devin work branch), whose
`arc-agi-3/submit_best.py` still pins:
- AGI-3 job → `takumuhata/forge-pathfinder-bfs-agent` (scored 0.08–0.23 — the
  Forge v7 submissions of 9/25–29 were this bug, not a score-aware pick)
- AGI-2 job → `arc-agi2-lb33-perfpatch-dsl` (28.06–29.72) instead of the plain
  `arc-agi2-lb33-89-perfpatch` (30.56 × 3, deterministic — verified byte-identical
  to mikelou1's public 33.89-named kernel; the ~3pt gap to 33.89 is eval-set
  variance, not a regression in our copy)

**Fixes applied**:
- PR #6 `devin/fix-submit-best-plain` → master: ports work-branch submit_best.py
  + kernel_versions.json only (65/-29 LOC). Merging restores correct picks.
- Automation prompt updated (pending user approval): adds
  `git fetch origin devin/1790016368-summary-txt-always && git checkout FETCH_HEAD`
  before running the scripts, so ticks use the work branch even while master lags.

**Data note**: LoRA forensic correction — SFT data is reasoning-DENSE (0% truly
bare assistant msgs; median 1310 chars of reasoning_content). Earlier "58.6%
empty tool_call" claim measured `content` only. Adapter was born-destructive
(delta norms ~3x pilot from ck37), cause still = recipe-scale overfit, not
missing reasoning. Retrain needs gentler recipe + GPU units.

## 2026-10-01 — Colab T4 eval of the production AGI-2 pipeline (VERIFIED RECIPE)

Goal: run the production kernel's per-task TTT + turbo-DFS decode + selection
locally on Colab Tesla T4 (14.5 GB, no bf16) for a score baseline without
burning Kaggle quota. Notebook: Drive `colab_ag2_eval.ipynb`
(id 16RXdDHX2TjzN14_aC_wHEPpFotnpiH7o). All execution is done via uploaded
files + Terminal/cells — do NOT re-run the original notebook cells
(they are the bf16/L4 config and will OOM).

**Working /content layout** (files uploaded one at a time via Files panel):
`arc_loader.py, arc_decoder.py, arc_solver.py, eval_starter.py, runeval.sh,
mon.sh, data/{challenges,solutions}.json` (+ `inference_outputs/`,
`eval_stdout.log`, `worker0`).

**T4 recipe — every one of these was needed, each removed a distinct failure:**
- `load_in_4bit=True` → model 1.92 GB (probe-verified); bf16 load is ~8 GB
- `fp16=False, bf16=False` in train_args → no AMP; fp16 GradScaler crashes on
  4-bit params ("Attempting to unscale FP16 gradients")
- NO fp16 param cast loop (it manufactures fp16 grads → same scaler crash)
- `use_gradient_checkpointing="unsloth"` (in from_pretrained AND get_peft_model)
  + `gradient_checkpointing=True` — plain True never engaged (~10 GB activations)
- `optim="adamw_8bit"` — adamw_torch ~2 GB vs ~0.5 GB
- `max_seq_length = 2048` — but see cut_to_len hazard below
- `target_modules` must NOT include `embed_tokens`/`lm_head`: unsloth's
  `offload_input_embeddings` torch.saves the embedding into the model dir
  (`/kaggle/input` is READ-ONLY → `Read-only file system` RuntimeError)
- arc_loader.py guard: `test = train[-1] if train else None` — `cut_to_len`
  drops train examples whose prompt exceeds max_len; at seq 2048 some tasks
  end with train=[] → `fmt_train` IndexError without the guard
- Peak resident during TTT ≈ 10.3-10.7 GB — fits with ~3.8 GB headroom
- Model load transient reaches ~10.4 GB (bf16 chunks quantize on GPU) — survives
- Per task: TTT (16-augment, r256, 1 epoch) then turbo-DFS decode over 16 aug
  variants → `.ex` pickles in /content/inference_outputs + per-candidate
  augmented scoring. End-to-end confirmed on task 0934a4d8.

**Execution channel gotchas (learned the hard way):**
- Files-panel upload = the only reliable file channel; GTK picker Ctrl+L then
  full path; multi-select uploads deliver only the first file — upload singly
- Terminal `type` drops `_`, `?`, `&&` → never type commands; use uploaded .sh
- Cell typing drops leading chars sporadically → start cells with a junk line
  (`x=1`) and avoid `!` shell lines (prefer `subprocess.run`)
- errored cells sometimes refuse further edits — add a new cell (Esc,B)
- stray file pickers steal typing — Escape them first

**T4 eval run result (Oct 1, ~1h45m for all 10 tasks):**
- All 10 tasks processed end-to-end (`[Rank 0] done!`, clean exit). Per-task
  289-728s — much faster than feared because decode exhausts early when
  nothing parses.
- **Local score: 0.0%** — only 2/10 tasks emitted any `.ex` pickles
  (0934a4d8: 5 candidates, 142ca369: 3), 0/9 candidates correct,
  ×120 extrapolation 0.0. 8/10 tasks produced ZERO parseable candidate grids.
- `training_loss=nan` in the log is a REPORTING ARTIFACT: train_args set
  `logging_strategy="no"` → empty log history → aggregate NaN regardless of
  the real loss. Not evidence TTT diverged.
- Decode-candidate yield ~5% of augmented variants vs healthy production —
  consistent with 4-bit quant + max_seq_length 2048 (cut_to_len drops long
  prompts) degrading generation quality for this structured task.
- VERDICT: **T4 (free Colab tier) is not a viable proxy for the L4/bf16
  pipeline.** Mechanics verified end-to-end, but the score signal is ~0 —
  too lossy to measure real changes. Baseline comparisons stay on Saturday
  Kaggle GPU quota.
- Scorer on Colab: `python3 /content/score.py` (uploaded via paste.rs —
  cell typing corrupts >300-char payloads; Kaggle dataset pull 403s on
  private sets with the KGAT key).
- arc_solver.py on /content has the left-pad fix for ragged decode batches
  (`prefix_tokens` padded to max len with PAD_ID=13); mirrored locally.
- Ablation: ran task 135a2760 with `trainer.train()` skipped (zero-init LoRA)
  → still ZERO candidate files in 90.9s. TTT is not the limiter; the 4-bit
  base model's raw generation is. Confirms the T4-free-tier verdict.
  Runtime disconnected/deleted after run (~5 units left on account).

## AGI-3 eval wall-clock analysis (Oct 1, /home/ubuntu/af_out = 8.21-run artifacts)

**Finding: the public-25 eval is wall-clock-bound, not step-bound.**
- Settings: `max_runtime_s_per_game=7920s`, `concurrency=28`,
  `analyzer_timeout=1200s` (mirrors competition; 9h kernel budget fits
  ~110 hidden games in 4 waves of 28).
- All 25 games ran concurrently and died together at global budget end —
  every non-completing transcript ends on a vLLM read-timeout (the final
  in-flight request truncated at remaining budget).
- Per game: ~50 analyzer calls, median 194s/call (mean 328s), median ~2k
  generated tokens/call → ~8.7 tok/s per stream (28 streams saturate the
  GPU; aggregate ~240 tok/s). **Turns are the currency; tokens/turn is the
  only lever.**
- Zeros/stuck games (bp35, g50t, sc25) burned the whole budget on LEVEL 1.
  ACTION7 was a required mechanic in ≥6 public games (rotation etc.) —
  calls 73-370/game, all rejected pre-fix. v2 bundles carry the fix.
- Late bloomers exist (tu93, dc22 first completions at ~85-89% of budget)
  → do NOT kill "hopeless" games early; it would cull real completions.
- Concurrency/budget are forced by 110 games / 9h — not free knobs.
  `LOCAL_ANALYZER_MAX_OUTPUT` hard cap left at 6144: truncation mid-tool-call
  wastes the whole turn — prompt-level brevity is the safer route.

**Saturday A/B (both pushes carry the same patch set: ACTION7+UNDO fixes,
perception hints, scoring line):**
- `arc3-duck-anim-v2` → `taaf-anim-flashnext-bundle-v2` — control arm.
- `arc3-duck-anim-flashnext` → `taaf-anim-flashnext-bundle-v3` (new,
  repo `bundle_fast/`) — experimental arm: +3 prompt hints: (1) compact
  reasoning / reserve long analysis for new level or failed plan,
  (2) early RESET on wedged push-puzzle states (bp35-style deadlocks),
  (3) finish understood levels in code — parse grid, compute sequence
  programmatically, batch actions (ft09 47.6 pattern).
- Push gating: `push_pending.py` now pushes ONLY on Saturday (UTC) —
  weekly GPU budget shared with RSNA (Sundays reserved). Bypass with
  `TAAF_PUSH_ANY_DAY=1`.
- Read-out: same public-25 → compare mean vs 8.21 baseline and vs each
  other; submit_best.py picks verified-mean ≥4.0 automatically.

**submit_best hardening (Oct 2):** Devin Review findings addressed —
(a) status=COMPLETE now submits kernel_version="latest" (the run that
produced the verified output) instead of the vermap pin, which drifted
stale whenever a version was pushed outside push_pending.py;
(b) both Saturday kernels write /kaggle/working/variant_id.json
(anim-v2 / flashnext-v3) so submit_best logs which code produced a run's
output (informational; no gating yet).

## Colab full-fidelity eval runbook (Oct 2, verified working)

Colab G4 (Colab Pro + PAYG) = RTX PRO 6000 Blackwell, sm_120, 96GB —
identical shape to the Kaggle NvidiaRtxPro6000. The full AGI-3 stack
serves natively; 25-game offline eval runs end-to-end with zero Kaggle
GPU quota. Cost: only compute units (G4 ~19.5/hr — the 135GB model
download eats ~1h of that, plus ~2.5-4h eval).

**Layout:** `/kaggle` on Colab is a read-only ext4 mount —
`mount --bind /content/kroot /kaggle` makes the whole kaggle tree
writable (run once per runtime; survives for the session). All driver
steps are idempotent.

**Auth (IMPORTANT — stale secret):** the `KAGGLE_CREDENTIALS` Colab
secret holds two OLD kaggle.json-style creds; they authenticate but
CANNOT see takumuhata's private datasets (`datasets list -m` empty,
401 on dataset status) — scoped/dead keys. Working creds are the kaggle
2.x OAuth pair on the ops VM (`~/.kaggle/access_token` +
`~/.kaggle/credentials.json` — credentials.json has the refresh token;
access_token alone is NOT enough). Inject once per runtime via the
Colab **Terminal** (not a cell — keeps the token out of the saved
notebook):
```
echo <access_token> > /root/.kaggle/access_token
echo <credentials.json-base64> | base64 -d > /root/.kaggle/credentials.json
```
Use `kaggle` >=2.x everywhere (1.x ignores access_token and the dead
kaggle.json keys). kaggle 2.x notes: `competitions download` has no
`--unzip` — it drops a single <comp>.zip, unzip manually; `datasets
download --unzip` unchanged; `kaggle whoami` is not a 2.x command.
kagglehub.model_download works unauthenticated for the public
keithtyser model.

**Driver:** `takumuhata/taaf-ag3-eval-src` dataset →
`colab_driver.py` + both eval notebooks. `TAAF_EVAL_VARIANT=v3|v2`
picks the arm (v3=taaf-anim-flashnext-bundle-v3+arc3-duck-anim-flashnext,
v2=taaf-anim-flashnext-bundle-v2+arc3-duck-anim-v2). It: apt ninja →
uv venv py3.12 → pip papermill+kaggle>=2+kagglehub → makes
/kaggle/input dirs → competition download+unzip → bundle+runtime
datasets → kagglehub model (~50min at ~38MB/s) → symlink MODEL_PATH →
papermill the eval notebook (cwd=/kaggle/working) → parse score.json →
upload score/transcripts/provenance to `takumuhata/taaf-ag3-eval-out`.

**Results flow:** MEAN line printed in cell output; artifacts also
land in taaf-ag3-eval-out for retrieval from the ops VM.

## Colab A100 reality check (Oct 2)

- This account (Colab Pro, pay-as-you-go units) gets **A100-SXM4-40GB max** — no 80GB A100, no G4 (Pro+ only). Confirmed via torch.cuda.get_device_name: `NVIDIA A100-SXM4-40GB`, 42.48GB VRAM.
- Consequence: 27B bf16 (54GB) and Qwen3.8-Flash-Next (~250GB MoE / ~65GB GPTQ) **cannot be served on Colab**. The AGI-3 v2-vs-v3 prompt A/B (`colab_ab_eval.ipynb`, commit d1e956b) is therefore blocked on this tier — it would need Pro+/G4 or Kaggle's RTX Pro 6000 (i.e. Saturday quota).
- What DOES fit: the AGI-2 production pipeline — 4B bf16 (~8GB) + unsloth LoRA + turbo-DFS. Faithful local eval is possible.

## AGI-2 faithful Colab eval (running Oct 2)

`submit_ag2_perfpatch/colab_ag2_eval_v3.ipynb` (commit f4a1c25): production perfpatch cells verbatim (arc_loader/arc_decoder/arc_solver) + Colab patches + single-GPU starter on 12 public-eval tasks.

- Model: `sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1` via `kagglehub.model_download` → `/root/.cache/kagglehub/models/...` — pass through `os.environ['MODEL_DIR']`, NOT the hardcoded `/kaggle/input/models/...` (that path only exists on Kaggle).
- Tasks: `takumuhata/arc-ag2-eval-json-public` (120 challenges+solutions, made public for anonymous fetch). **kagglehub `dataset_download` 403s on GetDataset even for public datasets on this runtime** — use plain `wget https://www.kaggle.com/api/v1/datasets/download/<slug>` instead.
- arc_solver.py needs THREE path patches: `model_name` → `os.environ["MODEL_DIR"]`, `dir_outputs` → `/content/inference_outputs`, and `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/` → `/content/data/` (the worker reads the challenges JSON itself — easy to miss; first run died on exactly this).
- Subset: kernel's own 4 eval-gate tasks + 8 evenly-spread tasks (269e22fb, 3e6067c3, 5dbc8537, 7b80bb43, 9385bd28, b5ca7ac4, dbff022c, e3721c99) = 12 tasks, ~1.5-3h on A100, ~20-40 units.
- Scoring: production `data.validate_submission` — local score is directly LB-comparable (120-task eval set; subset is a noisy proxy).
- Colab GUI gotcha, confirmed again: multi-line cell edits corrupt text (leading chars dropped mid-cell). Edit the .ipynb locally and re-upload instead — safer than fighting the editor.

## Where this leaves the loop

- AGI-2: real local signal → can A/B test variants (perfpatch vs nvarc vs param tweaks) on Colab without Kaggle quota.
- AGI-3: no sub-80GB proxy that preserves score signal (T4-4bit=0 signal, A100-40 can't fit 27B+). The Saturday Kaggle runs remain the only faithful eval — prompt A/B rides on the queued anim-v2/flashnext-v3 pair.

### Tokenizer dead-TTT bug (Colab-only) + production verified alive (Oct 2)

- The shrunk grids15 tokenizer is `WordLevel` (tokenizer.json): vocab has
  'user'=11, 'assistant'=12, but the pretokenizer never emits them under
  transformers 5.5.0/tokenizers-new → `encode('user')`→`[]` → collator's
  np.where(USER/ASSISTANT) empty → ALL labels -100 → TTT loss=nan,
  grad_norm=0 (complete no-op) on the Colab stack.
- Fix (verified): before training, register them as specials —
  `tokenizer.add_tokens([AddedToken("user", special=True, normalized=False),
                        AddedToken("assistant", special=True, normalized=False)])`
  → encode emits [14,11,10,...,14,12,10] as intended.
- **Production is NOT affected**: perfpatch kernel log (downloaded via
  `kaggle kernels output`) shows real `training_loss` 0.005→0.0001 at
  global_step=128 on every task. Kaggle stack = transformers 4.55.4 /
  unsloth 2025.9.7 / torch 2.8.0+cu128 on L4×4. LB30.56 already includes
  working TTT — the fix is required only for Colab eval faithfulness.
- Production candidate yield (submission.json audit): 4/120 tasks emit
  real grids; the other 116 emit `[[0]]` placeholders → LB 29.72 comes
  almost entirely from ~4 solved tasks. Even +1 task ≈ several points —
  candidate yield is THE lever.

## Colab prompt A/B — v2 vs v3 brevity hint (Oct 2, T4 + Qwen3-4B thinking)

- Proxy test: does the v3 prompt bundle's compact-reasoning line reduce
  generated tokens/turn (the score-binding lever — median 2037 tok/turn,
  ~8.7 tok/s/stream → tokens/turn is the only controllable budget)?
- Setup: Qwen3-4B bf16 on T4, enable_thinking=True, temp 0.6 top_p 0.95
  top_k 20, max_new 3072; 3 reconstructed scenarios (cold L1, stuck L2
  sokoban, late L4 w/ 850s left); sys prompts = real assembled
  _build_system_prompt output, v2 vs v3 (diff = the 3 v3 lines).
- RESULT: v3 did NOT reduce tokens. gen_tokens v2=[910,928,1405]
  v3=[1029,1637,1209] (mean +20%); think_tokens v2=[519,798,1259]
  v3=[628,1431,999]. Worst in stuck_L2 (1637 vs 928) — the RESET hint
  may have *added* deliberation. tool_call format 0/3 both variants;
  action() text refs 2-3/3 both.
- Follow-up: v4 = v3 + hard cap "assistant text must stay under 120
  words before next tool call" — same-protocol check on stuck_L2.
- Interpretation guardrails: 4B proxy vs 27B production, 1 seed/scenario;
  a null here does not kill the v3 arm for Saturday (model families and
  scale differ), but weakens the brevity-hint hypothesis — the A/B on
  the real model stays the arbiter.

### v4 hard-cap follow-up (Oct 2) — applied to Saturday experimental arm

- stuck_L2 single-scenario rerun, same protocol: v2 gen=1481/think=1293
  (242.7s) vs v4 (v3 + "assistant text must stay under 120 words before
  next tool call") gen=848/think=742 (148.1s) → ~43% fewer tokens and
  output went straight to `python`/`action()` code.
- Cross-run pattern: soft "keep compact" hint does not move tokens
  (v3 even inflated stuck_L2 to 1637); an explicit word cap does.
- Applied: prompts.py in bundle_fast gained the HARD CAP line ahead of
  the soft hint; taaf-anim-flashnext-bundle-v3 re-uploaded (v4) and
  HARD CAP verified in the served file. Saturday experimental arm now
  carries the capped variant; control arm unchanged.
- Sampling-variance note: single seed per scenario, v2 stuck_L2 itself
  swung 928→1481 across runs — treat as directional, not conclusive.

### Kaggle API code-submission block (Oct 2) — NEW BLOCKER + workaround

- `CreateCodeSubmission` now returns 403 `Permission 'kernelSessions.get'
  was denied` on EVERY credential: admin-scoped OAuth token (fresh refresh),
  classic KAGGLE_API_TOKEN, and after removing stale access_token file.
  Kernel status/output reads still work; AGI-2 file submissions still work
  (this is a CODE-submission-only block).
- This matches a Feb-2026 report that Kaggle is restricting code-competition
  submissions via public API — enforcement apparently reached this
  account/competition between Oct 1 05:02 (last working submit) and Oct 2.
- WORKAROUND THAT WORKS: browser UI submit. Chrome is logged into Kaggle
  via Google account thatakumu@gmail.com (= takumuhata). Path:
  competition page → "Submit Prediction" → Notebook tab → pick kernel +
  version + submission.parquet → Submit. Verified live: anim-flashnext V1
  submitted Oct 2 ~11:15 UTC (status "Notebook Running").
- CONSEQUENCE: automation ticks can still run submit_best (AGI-2 file
  submits keep working; AGI-3 attempts now log a 403). AGI-3 daily
  submissions need the browser path — either driven from a session with
  Chrome access or manually by the user.
- Saturday queue unchanged: kernel PUSHES use a different endpoint
  (PushKernel) — no evidence they are blocked; verify on first push.

### AGI-2 medal push plan (Oct 1) — evalscan kernel

Goal: +2.2 pts to reach ~top-40 territory (best 30.56, top40=32.78).

Pipeline recap (arc-agi2-lb33-89-perfpatch, LB 30.56): per task → LoRA
TTT (r256, 1 epoch on 16 augmented train samples) → turbo-DFS decode,
8 subkeys per test input grouped into 4 batches (rotation family then
transpose family), spend cap 1200s/task, DFS inner cap 540s, global
12h-10min budget, L4x4 workers. Selection = score_kgmon vote count +
aug NLL → top-2 unique grids = attempt_1/2.

Measured this session:
- dsl_all battery (31 solvers): 0/120 eval tasks, 0/172 test inputs.
  Symbolic fallback contributes NOTHING on eval; the -dsl variant's
  -0.83pt was pure displacement (DSL preds injected at attempt_1).
  Do not retry DSL injection except strictly into attempt_2-and-only-
  when-no-model-candidates — and even then expected gain ≈ 0.
- Failure-mode question reframed: not yield (4-task submission.json is
  a normal-run artifact; rerun processes all 120 test tasks), but
  which tasks produce correct candidates and why the rest don't.

evalscan kernel (takumuhata/arc-agi2-evalscan, submit_ag2_evalscan/):
- Identical perfpatch code; starter.py whitelist removed so a NORMAL
  run decodes all 120 evaluation tasks. Rerun path untouched.
- Saves /kaggle/inference_outputs pickles into /kaggle/working (kernel
  output), computes dsl_preds.pkl (analysis only, NOT injected), prints
  per-task decode-coverage diagnostic + benchmark_selection_algos.
- Answers on one ~10-12h L4x4 run: (a) do tail tasks starve under the
  1200s×120/4 = 10h budget? (b) oracle headroom — how many test inputs
  have the correct grid anywhere in candidates? (c) which selection
  algo is best locally? (d) beam-score gap correct-vs-wrong.
- Offline follow-up: arc-agi-2/dev/select_bench.py loads the saved
  pickles + eval solutions → scores kgmon / probmul_3 / aug-family-
  diverse attempt_2 / oracle. Decode is the expensive part; selection
  is free — one run evaluates every strategy.

Saturday (UTC) GPU order: evalscan first (highest information), then
queued AGI-3 pushes anim-v2 (control) + anim-flashnext (v4 HARD CAP).
If quota/concurrency won't fit all three, AGI-3 pushes slip — the
A/B experiment can wait a week; the medal math can't.

### Oct 2 afternoon — AGI-2 variant staging + AGI-3 v5

- evalscan hardened: dir_outputs -> /kaggle/working/inference_outputs
  (pickles persist even if run times out); tlog() writes TTT/batch/total
  timing to /kaggle/working/timing.log; analyze_evalscan.sh = download +
  timing summary + select_bench in one command.
- faircap fixed: queue is shared by 4 workers -> fair share is
  4*remaining/qsize (was 4x too tight).
- submit_ag2_xcalibur staged (sft139 -> xcalibur-aa2-sft-500 drop-in);
  hold until evalscan shows whether failures are capability-bound.
- AGI-3 v5 prompt staged in bundle_fast (PERSISTENT NOTES via `result`
  echo — zero-plumbing cross-turn memory). NOT uploaded to the v3
  dataset: Saturday's arm still tests v4 HARD CAP cleanly.
- Dead ends re-measured on eval: dsl_all 0/120 tasks; identity golds
  0/172; rigid-geo golds 0/172 — symbolic fallback fully dead.
- Automation dispatch failed 3x today (ACU); manual path verified —
  run submit_best.py from main session if Sat 04:45 tick fails too.

### Oct 3 (Sat) — 05:00 wake: manual queue run + API block lifted

- Automation dispatch failed a 4th time at 04:45 UTC; ran the queue
  manually from the main session as planned.
- AGI-3 A/B pushed: arc3-duck-anim-v2 v2 (control arm) QUEUED +
  arc3-duck-anim-flashnext-v3 v2 (v4 HARD CAP arm) QUEUED. Both GPU
  batch slots occupied (~6-8h).
- SLUG CORRECTION: `arc3-duck-anim-flashnext` never existed — the real
  kernel is `arc3-duck-anim-flashnext-v3` (created Sep 21; v1 = the
  8.21-mean run we've been submitting). submit_best CANDIDATES +
  KNOWN_MEAN repointed to -v3. kernel_versions.json fixed. Metadata
  title in submit_ag3_hybrid set to resolve to the -v3 slug.
- Kaggle API code-submission block LIFTED: today's AGI-3 API submit
  succeeded (ref 56789981, -v3 v1, PENDING). No more daily UI submits
  needed unless it regresses. Yesterday's AGI-3 = 3.33; AGI-2 = 29.72.
- AGI-2 daily submitted via API: ref 56789960.
- evalscan push failed on "Maximum batch GPU session count of 2" —
  stays in pending_pushes.json; retry when a slot frees (~11-13 UTC).
  Kernel-metadata titles fixed to resolve to their ids (evalscan,
  faircap, xcalibur). One-time wake reminder set for ~11:30 UTC;
  if no slot by ~13:00 UTC, cancel anim-v2 (control — baseline 8.21
  already known) to start evalscan before the 15:00 UTC gate.

### Oct 3 — evidence from the whitelist run's log (free data)

- The 4-task whitelist kernel output keeps the real production log:
  per-task TOTAL 582/647/1103/1239s; TTT 208-828s (35-67% of task
  time — TTT, not decode, dominates; 128 steps).
- submission.json carries all 120 tasks (canned baseline + 4 fresh
  decodes). Vs public eval solutions: whitelist 3/4 correct
  (36a08778, 981571dc, aa4ec2a5; 0934a4d8 missed — correct grid never
  generated: capability miss, the xcalibur target class).
- The log emits ALL_CORRECT lines per decoded subkey (fires only when
  gold known): correct grid found in 12/16/8 aug variants on the 3
  solved tasks — beam scores range 0.97 down to ~0 (vote-count, not
  beam score, surfaces weak-but-correct candidates; kgmon works).
- analyze_evalscan.sh now aggregates ALL_CORRECT lines into a
  generation-vs-selection split (strong/weak/never-generated).
- TASK-CLASS hypothesis (whitelist n=1): eval output-shape classes:
  same 84, extraction (output smaller) 30, larger 6. The only
  whitelist miss (0934a4d8) is the only extraction task — if the
  class systematically fails generation that's up to ~25% of tasks
  (~7.5 LB pts) in play. analyze_evalscan now prints gen-rate per
  class; if 'extract' gen-rate ~0, target it with crop-aware decode
  or an extraction hint prompt (xcalibur probe: extend whitelist to
  evalscan's missed IDs).
- evalscan cell9 bug fixed: loaded decode results from the dead path
  /kaggle/inference_outputs (would crash -> skip cell10 analysis);
  now /kaggle/working/inference_outputs like cell10. Notebook is
  syntax-verified; starter queues all 120 eval tasks.
- submit_ag2_diverse2 staged (evalscan base + score_kgmon_diverse):
  attempt_1 = kgmon top-1, attempt_2 = best candidate whose dominant
  aug-family (mod chain minus permute*) differs from top-1's. Push
  it only if evalscan's select_bench shows diverse_rank2 > kgmon.
- AGI-3 landscape (tufalabs duck-harness writeup): our stack derives
  from this harness (their 27B-FP8 baseline: mean 1.60); public top
  = executable world models (RGB/OpenCode 58.12%, Symbolica 36.08%)
  — build a code simulator of game dynamics, plan internally, spend
  fewest real actions. Frontier-model capability dominates; within
  1-GPU constraints the levers are world-model prompting (v5 staged),
  action efficiency, exploration heuristics. Our 8.21 local mean
  already beats the reference harness by ~5x with same-class models.
- select_bench: per-class oracle + strategy scores (same/extract/
  larger) — total 172 test outputs (29 extract, 6 larger, 85 same).
- Mechanism note: subkey 'ex{N}' encodes WHICH train examples fit
  the context window (cut_to_len drops examples on long inputs).
  30x30 tasks (most extraction-type) decode with FEWER train
  examples -> compounding miss risk. Check ex-field diversity per
  task in evalscan subkeys; possible fix: more example-subset
  coverage per task, or raise max_seq_length on big-grid tasks.
- Starvation pre-estimate (no GPU): crude cell-cost model fit on the
  4 whitelist anchors -> ~6.1h/worker vs 11.8h budget (probably OK),
  BUT cost correlates weakly with actual time (0934a4d8 cost 4593 ran
  582s vs 36a08778 cost 3048 ran 1103s — decode difficulty, not grid
  size, drives duration). faircap cost now includes train outputs.

## 2026-10-06 — lane state rebuilt in devin-e6d7bb1298cb4814a0ccfe7b68c2fc44

Previous lane VM (session 514da5ce) is suspended; its local commits were
unreachable (repo archived, no push). Rebuilt this branch's tree from the
part attachments in that session's chat history:

- parts 1..14a + 14d applied on 259889c and committed one-per-part
  (see git log). File hashes verified against the part-14 sha table:
  sft_ag2.py 7566fae6, game_hints.py 8f695a49, check_patch_cell.py
  7540b772 (tail respliced from the resent 47 lines).
- medal_closeout.md: same xcalibur->NVARC sed repoint replayed.
- check_patch_cell.py: kept Claude's (20,5) fixture counts — this tree's
  submit_ag3_flashnext_patched ipynb is the pre-ops-fix version, so the
  (18,3) adjustment does not apply here.
- All tool --selftest pass (torch cpu venv: ~/.venv-cputorch).
- MISSING (then reconstructed): part 14b (make_colab_sft_nb.py,
  colab_sft_runbook.md, colab_sft_ag2.md update) and part 14c
  (wm_level2_only.py, wm_short_prompt.py) never reached either session.
  Reimplemented in this session from Claude's delivery-note index
  (559-line spec recovered from session 514da5ce attachments), NOT
  byte-identical to Claude's originals — same contract, verified by
  --selftest: notebook is 0.67 MB with 10 embedded files round-tripping
  byte-identically; wm_level2_only gates solve/explore/search/check/plan/
  health/execute to level>=2 in either apply order w.r.t. turn_budget;
  wm_short_prompt lands at exactly -1,043 prompt words / one 109-word
  bullet, matching the spec's numbers exactly. Commits below.
- SUPERSEDED: Claude then re-sent the real 14b/14c diffs (plus 14d,
  which was already applied + sha-verified). The reconstructions were
  replaced by his files; all 5 sha256 now match the handoff table
  (e659fc11/7d743da2/000b318c/6cccb399/e6594af0). colab_sft_ag2.ipynb
  regenerated by the real generator (0.58MB, 12 cells, 70/30 mix — note
  his default data share differs from my reconstruction). Reconstruction
  commits stay in history for traceability.

## 2026-10-06 — Lightning AI lane added (devin-e6d7bb)

make_lightning_sft_nb.py: Lightning Studio variant of the SFT notebook
(shared cells imported byte-identically from make_colab_sft_nb; Drive
mount replaced by the persistent Studio filesystem, credentials by
env-var/file fallback). Free tier ~80 GPU-hours to start; T4 ~75h
covers ~7 full runs. lightning_sft_ag2.ipynb is a generated artifact,
regenerate with the script. --selftest passes (Devin side, and re-run on
the local WSL clone ~/arc-prize-lane on 2026-10-07 together with
make_colab_sft_nb / wm_level2_only / wm_short_prompt selftests; the
regenerated ipynb differs from the committed one only in the gzip
header bytes of the embedded blobs (Python version); payload sha256s
are identical, committed file kept).
