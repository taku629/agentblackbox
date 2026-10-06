# Runbook: is checkpoint X better than sft139? (panel -> verdict)

All commands from the repo root on the ops VM. Building is local; pushing the
built directories stays inside the Saturday push window. `<cand>` is e.g.
`pranshubahadur/xcalibur-aa2-sft-500/Transformers/bf16/1`.

## 1. Gate the checkpoint (minutes, no GPU)

    kaggle models instances versions download <owner>/<slug>/transformers/<variation>/<n> -p /tmp/cand --untar
    kaggle models instances versions download sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1 -p /tmp/sft139 --untar
    python3 arc-agi-2/dev/check_swap_model.py /tmp/cand /tmp/sft139      # must print DROP-IN OK

## 2. Pick the panel once (from the evalscan pickles) and keep it fixed

    python3 arc-agi-2/dev/swap_eval.py panel /tmp/evalscan/inference_outputs --n 24 --timing /tmp/evalscan/timing.log
    # last line is:  --whitelist id1,id2,...   -> save it as PANEL

Use the SAME panel for every candidate and for the baseline, so runs stay comparable.

## 3. Build two kernels: baseline and candidate, same panel, both with the gold-NLL probe

    python3 arc-agi-2/dev/make_gen_kernel.py --probe-gold --whitelist $PANEL --name "ARC AGI2 Sft139 Panel"
    python3 arc-agi-2/dev/make_gen_kernel.py --probe-gold --whitelist $PANEL --name "ARC AGI2 Xcalibur Panel" \
        --model-source <cand>

No levers on either side: the comparison is the model, nothing else. `--name`
gives each its own slug and `submit_ag2_*` directory (without it the build
overwrites the genboost submission kernel). `--whitelist` also moves the
pickles to `/kaggle/working`, so they are in the kernel output.

Optional smoke before the panel (4 production tasks, ~1 h): the same command
without `--whitelist` and with another `--name`. Check in its log that
`allocated ...MB for training` and the per-task `finished ... in ...s` match
the baseline log; a quality reading from 4 tasks means nothing.

## 4. After both runs: download and compare

    kaggle kernels output takumuhata/arc-agi2-sft139-panel   -p /tmp/panel_base
    kaggle kernels output takumuhata/arc-agi2-xcalibur-panel -p /tmp/panel_cand
    python3 arc-agi-2/dev/swap_eval.py compare /tmp/panel_base/inference_outputs /tmp/panel_cand/inference_outputs \
        --base-nll /tmp/panel_base --cand-nll /tmp/panel_cand --name xcalibur

(`kernels output` serves only the newest version of a kernel -- download before re-pushing.)

Sanity check first: the baseline panel run should reproduce the evalscan
outcome on those 24 tasks. If `swap_eval.py compare /tmp/evalscan/inference_outputs /tmp/panel_base/inference_outputs`
shows gained/lost inputs, the pipeline is not deterministic enough for a
24-task verdict and the thresholds below need widening.

## 5. Verdict (printed on the last line)

| verdict | rule | next |
|---|---|---|
| promote | generated inputs: net >= +2 and lost <= 1 | full run: rebuild the candidate with `--whitelist all`, compare against evalscan |
| inconclusive | net +1, or median gold NLL better by >= 0.5 nats | larger panel (`--n 48`), same two builds |
| reject | anything else | drop the checkpoint; keep the gold-NLL delta as the measure of what 63 updates of SFT buy |

Also read, whatever the verdict:
- `gold NLL ... on the N inputs the baseline missed: median ...` -- how far a
  known amount of extra SFT moved the missed answers. This calibrates the
  bands in `gen_autopsy.py` section 4 and is the evidence for or against
  running `sft_ag2.py`.
- `lost:` ids -- regressions on tasks the baseline solved.

## 6. Adoption (only after a promoted full run)

A checkpoint replaces sft139 in the submission kernel only if the full
120-task comparison still shows net >= +3 generated inputs. Then:

    python3 arc-agi-2/dev/make_gen_kernel.py --verdict /tmp/evalscan/select_verdict.json --slack <SLACK> --model-source <cand>

(no `--whitelist`, no `--name`, no `--probe-gold`: this is the submission build).
