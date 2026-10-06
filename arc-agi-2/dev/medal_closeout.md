# Medal close-out runbook: Saturday outputs -> final submissions

One pass, top to bottom. Every box ends in exactly one build command or "keep".
Commands run from the repo root on the ops VM. Building is local; pushing and
submitting stay with the existing gate (`push_pending.py`, Saturday < 15:00 UTC).
Nothing here targets T4.

Numbers used below: AGI-2 best 30.56 (one public task = 0.83 points; top-40 ~32.78,
i.e. +3 tasks). AGI-3 best 3.98 hidden / notes arm 6.59-7.24 public-25 (run sigma
~0.85, so two arms differ for real only beyond ~1.7).

## 0. Download everything first (`kernels output` serves only the newest version)

    O=/tmp/sat; mkdir -p $O
    kaggle kernels output takumuhata/arc-agi2-sft139-panel   -p $O/panel_base
    kaggle kernels output takumuhata/arc-agi2-NVARC-panel -p $O/panel_nvarc
    kaggle kernels output takumuhata/<union panel slug>      -p $O/union_panel
    kaggle kernels output takumuhata/arc-agi2-genboost       -p $O/genboost
    kaggle kernels output takumuhata/<wm arm slug>           -p $O/wm
    kaggle kernels output takumuhata/<flashnext slug>        -p $O/flashnext
    # notes arm output already downloaded: NOTES=~/notes_out
    PANEL=<the id list printed by `swap_eval.py panel` and used for both panel builds (swap_runbook step 2)>

## 1. AGI-2

### 1.1 Is the panel usable at all? (2 min)

    python3 arc-agi-2/dev/swap_eval.py compare /tmp/evalscan/inference_outputs $O/panel_base/inference_outputs --name rerun

| output | meaning | do |
|---|---|---|
| gained 0, lost 0 (or 1 in total) | the pipeline repeats itself on 24 tasks | go on |
| more than 1 input moved | run-to-run noise is as large as the effect we look for | treat every `promote` below as `inconclusive`; only a full 120 run or the LB decides |

### 1.2 Model verdict (NVARC vs sft139)

    python3 arc-agi-2/dev/swap_eval.py compare $O/panel_base/inference_outputs $O/panel_nvarc/inference_outputs \
        --base-nll $O/panel_base --cand-nll $O/panel_nvarc --name NVARC | tee $O/v_model.txt
    python3 arc-agi-2/dev/swap_eval.py union $O/panel_base/inference_outputs $O/panel_nvarc/inference_outputs | tee $O/v_union.txt

Read the LAST line of each. `v_model` is one of `promote / inconclusive / reject`;
`v_union` starts with `build the union kernel` / `swap, do not union` / `no union`.

### 1.3 Does a second pass fit the hidden rerun? (needs the union panel log)

    python3 arc-agi-2/dev/rerun_sim.py $O/panel_base/timing.log --tasks 240 | tee $O/v_time.txt
    grep -c "pass 2: done" <union panel kernel log>     # must be 4 (one per worker); 0 = the model swap failed
    grep "pass 2: loading" <union panel kernel log>     # shows the seconds left when model B was loaded
    ls $O/union_panel/inference_outputs | grep -c runm2   # must be > 0

Already measured on the evalscan timing: 240 tasks all reached, 9.6 worker-hours idle
under the 1200 s cap. If the three checks above hold, time is not the constraint.
If `pass 2: done` is missing, the union kernel is NOT a candidate this week,
whatever `v_union` says (loading a second model in one process was never run on GPU before this panel).

### 1.4 Decision table -> the production build

Production build = no `--whitelist`, no `--probe-gold`, no `--name` (so it lands in
the submission directory). `LEVERS` = the genboost decision if the genboost LB
score is >= 30.56, else `{}` (see 1.5).

| v_union | v_model | pass-2 checks (1.3) | build | submit slug |
|---|---|---|---|---|
| build the union kernel | any | ok | `python3 arc-agi-2/dev/make_union_kernel.py --second takumuhata/qwen-nvarc/Transformers/bf16/1 --levers "$LEVERS"` | `takumuhata/arc-agi2-union` |
| build the union kernel | promote | failed | `python3 arc-agi-2/dev/make_gen_kernel.py --levers "$LEVERS" --model-source takumuhata/qwen-nvarc/Transformers/bf16/1` | `takumuhata/arc-agi2-genboost` |
| swap, do not union | promote / inconclusive | - | same `make_gen_kernel.py ... --model-source` line | `takumuhata/arc-agi2-genboost` |
| no union | promote | - | same `make_gen_kernel.py ... --model-source` line | `takumuhata/arc-agi2-genboost` |
| no union | inconclusive / reject | - | keep sft139: `make_gen_kernel.py --levers "$LEVERS"` if LEVERS is not `{}`, else keep `takumuhata/arc-agi2-lb33-perfpatch` | genboost or perfpatch |
| build the union kernel | inconclusive / reject | failed | keep sft139 (as the row above) | genboost or perfpatch |

With `LEVERS='{}'` and no model swap there is nothing to build: keep perfpatch
(`make_gen_kernel.py --levers '{}'` would only produce a copy of it under the genboost slug).

Before pushing any build: the directory's `kernel-metadata.json` must say
`"machine_shape": "NvidiaL4"` and list the model(s) you expect in `model_sources`.

    python3 -c "import json,sys; d=json.load(open(sys.argv[1]+'/kernel-metadata.json')); print(d['id'], d['machine_shape'], d['model_sources'])" submit_ag2_union

### 1.5 genboost (levers) verdict

The genboost run is a submission kernel; its evidence is the LB score.

| genboost LB | do |
|---|---|
| >= 31.39 (30.56 + one task) | levers ON in the production build: copy the dict from `grep -o "GEN_LEVERS = {[^}]*}" submit_ag2_genboost/*.ipynb`, keep only the entries that are not off, write it as JSON into `LEVERS` |
| 30.56 +- one task | levers are neutral; keep them only if the union/swap build needs faircap (`{"faircap": true}`), otherwise `LEVERS='{}'` |
| <= 29.73 | `LEVERS='{}'` |

(Lever keys: `g1` (`"off"` or a mode such as `"empty_input"`), `g2`, `g3`, `faircap` (true/false).
The union builder turns `faircap` on by itself.)

### 1.6 Final two selections (Kaggle counts the best of the selected ones on the private set)

1. the highest public-LB kernel run so far (today: perfpatch 30.56 unless Saturday beat it);
2. the most DIFFERENT one that is not worse by more than one task: union > swapped model > genboost > perfpatch.
Never select two runs of the same kernel.

### 1.7 Next Saturday's panels (build today, push in the next window)

Ranked third/fourth model candidates (all checked with `check_swap_model.py` on the
real config/tokenizer/safetensors headers; weights compared on 5 tensor slices):

| # | checkpoint | where | verdict | distance to sft139 (rel. L2) | note |
|---|---|---|---|---|---|
| 1 | `iamPi/Qwen-NVARC` | Hugging Face only | DROP-IN OK, 1 WARN (tied embeddings) | 0.11-0.22 | the only different training run; no licence file; must be mirrored to a Kaggle Model |
| 2 | `sorokin/qwen3_2b_grids15_sft141/Transformers/bfloat16/1` | Kaggle Models | DROP-IN OK, WARN (2B: 1.41 B params) | different size | ~0.39x time per task; Apache 2.0 |
| 3 | `konstantinboyko/qwen3-4b-bfloat16-v02-01-01/Transformers/default/1` | Kaggle Models | DROP-IN OK | 0.0000-0.0006 | closer to sft139 than NVARC (0.0002-0.0005): expect the NVARC verdict or less |
| - | `maximecarriere/qwen3-8b-arc-custom-merged` | Kaggle Models | NOT a drop-in | - | 8B, 16 ids but `<|im_start|>user` is ONE token (id 11) and id 14 is `<unk>`; 12.9 GiB fails the L4 TTT gate |

    # 1: mirror + gate (licence question to settle first, see the hand-over note)
    huggingface-cli download iamPi/Qwen-NVARC --local-dir /tmp/nvarc
    python3 arc-agi-2/dev/check_swap_model.py /tmp/nvarc /tmp/sft139          # DROP-IN OK expected
    #    upload as a Kaggle Model exactly as in colab_sft_ag2.md "upload" (owner/slug/Transformers/bf16/1), then
    python3 arc-agi-2/dev/make_gen_kernel.py --probe-gold --whitelist $PANEL --name "ARC AGI2 Nvarc Panel" \
        --model-source <owner>/qwen-nvarc/Transformers/bf16/1
    # 2: no upload needed
    python3 arc-agi-2/dev/make_gen_kernel.py --probe-gold --whitelist $PANEL --name "ARC AGI2 Sft141 2b Panel" \
        --model-source sorokin/qwen3_2b_grids15_sft141/Transformers/bfloat16/1

Then 1.2 again per candidate (`compare` + `union` against `$O/panel_base`). The
candidate with the best `union` line becomes `--second` in 1.4; there is no
three-pass kernel, so a third model replaces the second, it is not added.

## 2. AGI-3

### 2.1 Autopsy both arms (5 min, CPU)

    python3 arc-agi-3/dev/ag3_autopsy.py ~/notes_out --label notes --transfer 0.48 --json $O/notes_autopsy.json | tee $O/a_notes.txt
    python3 arc-agi-3/dev/ag3_autopsy.py $O/wm       --label wm    --transfer 0.48 --json $O/wm_autopsy.json    | tee $O/a_wm.txt
    python3 arc-agi-3/dev/ag3_autopsy.py ~/notes_out --label notes --compare $O/wm --no-games | tail -40

First line of each report must show `harness reported` equal to the mean it
recomputed; if not, stop and send the report (a file was not parsed as assumed).
`--transfer 0.48` is hidden/public for the one kernel with both numbers (3.98 / 8.21).

### 2.2 Which arm is the daily submission

| wm arm mean vs notes (6.9) | submit daily | follow-up build |
|---|---|---|
| >= 8.1 | wm arm | 2.3 with the wm report |
| 5.7 .. 8.1 | the higher of the two; if within 0.5, notes (fewer moving parts) | 2.3 with the wm report |
| <= 5.7 | notes | `wm_modes.py --mode toolbox` on a copy of the search-level bundle |

flashnext: submit it instead only if its public-25 mean beats the chosen arm by more than 1.7.

### 2.3 Next build from the autopsy VERDICT line (largest bucket)

| largest bucket | what the table shows | build |
|---|---|---|
| model (`no_mechanics`, `wm_diverged`, `wm_drop`) | enough effective actions, mechanics still not understood | wm arm: `wm_modes.py --mode toolbox` if `wm_drop`+`wm_diverged` cover more than half of the games, else `--mode full` unchanged and spend the next run on the notes prompt |
| explore (`noop_waste`, `loop`) | actions that change nothing or revisit boards | wm bundle with `wm_round3.py --turn-planner` (explore before exploit) |
| budget (`starved`) | fewer frontier actions than a human needs | turn planner as above; raise actions per turn before anything else |
| goal (`goal_unfired`) | model is accurate, completion test never fires | `wm_search.py` + `wm_modes.py --mode full` (search toward `progress`) |
| efficiency | levels are completed but with too many actions | nothing to build: points are in more levels, see the TO REACH block |
| infra (`crashed`) | runs died | fix first; nothing else is measurable |

Apply order on a fresh copy of the bundle, always:
`apply_wm_patch.py` -> `wm_round2.py` -> `wm_round3.py [--turn-planner]` -> `wm_search.py` -> `wm_modes.py --mode full|toolbox`.
`--mode toolbox` cannot be turned back into `full` on the same tree; keep two copies.

### 2.4 What the medal line means (scoring rule only, public-25 baselines)

    python3 arc-agi-3/dev/ag3_autopsy.py --structural

| your actions / baseline | level score | levels needed for 30 (best case) |
|---|---|---|
| 1.00x | 100 | 47 of 183 |
| 1.25x | 64 | 74 of 183 |
| 1.50x | 44 | 113 of 183 |
| >= 1.83x | <= 30 | unreachable even with every level completed |

So 30 on the public set needs roughly a quarter of all levels at human efficiency,
and the hidden score has so far been about half the public one. Use the report's
`TO REACH 30` block for the exact count from the current run.

## 3. Hand-over notes (things only a person can decide)

- `iamPi/Qwen-NVARC` has no licence file and no model card. Using it means mirroring it to Kaggle;
  whether that is acceptable under the competition's external-model rule is a call for the account owner.
- Everything in section 1 except the panel comparison itself is unverified on GPU until Saturday's logs say otherwise;
  the checks in 1.3 are the gate.
