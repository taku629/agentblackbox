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
