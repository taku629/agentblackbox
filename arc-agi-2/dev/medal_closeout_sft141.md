# medal_closeout.md addendum: the cheap union (sft139 -> sft141 2B) when the NVARC panel does not promote

Paste as section 1.4b of `medal_closeout.md` (kept as a separate file because that runbook was
edited on the ops side after part 11). `$O`, `$PANEL` and `$LEVERS` are the variables of sections 0 and 1.5.

Why this arm exists: `sorokin/qwen3_2b_grids15_sft141/Transformers/bfloat16/1` is a DROP-IN
(16-token vocabulary, 1.41 B parameters, Apache 2.0, already on Kaggle Models) and costs about 0.39x
the time of a 4B pass. A weaker second model can still add inputs the first one never generated, and
a pass that cheap fits the hidden rerun even if pass 1 runs long.

## When to use it

| NVARC panel (`v_model`) | NVARC union (`v_union`) | NVARC pass-2 checks | next |
|---|---|---|---|
| promote | build the union kernel | ok | NVARC union as in 1.4 -- this addendum is not needed |
| inconclusive / reject | no union | - | build the sft141 panel below |
| any | build the union kernel | failed (no `pass 2: done`) | fix the two-model load first; the sft141 union uses the same code path and would fail the same way |
| inconclusive | build the union kernel | ok | NVARC union stays the candidate; run the sft141 panel in the same window as the second option |

## Panel (next push window)

    python3 arc-agi-2/dev/make_gen_kernel.py --probe-gold --whitelist $PANEL --name "ARC AGI2 Sft141 2b Panel" \
        --model-source sorokin/qwen3_2b_grids15_sft141/Transformers/bfloat16/1
    #   -> submit_ag2_sft141_2b_panel/ , slug takumuhata/arc-agi2-sft141-2b-panel
    kaggle kernels output takumuhata/arc-agi2-sft141-2b-panel -p $O/panel_2b
    python3 arc-agi-2/dev/swap_eval.py compare $O/panel_base/inference_outputs $O/panel_2b/inference_outputs \
        --base-nll $O/panel_base --cand-nll $O/panel_2b --name sft141 | tee $O/v_model_2b.txt
    python3 arc-agi-2/dev/swap_eval.py union $O/panel_base/inference_outputs $O/panel_2b/inference_outputs --name sft141 | tee $O/v_union_2b.txt

`compare` is expected to say `reject` (a 2B model alone is not better than the 4B one); that is not
the question. Only the last line of `v_union_2b.txt` matters.

## Decision

| `v_union_2b` last line | build | submit slug |
|---|---|---|
| build the union kernel | `python3 arc-agi-2/dev/make_union_kernel.py --second sorokin/qwen3_2b_grids15_sft141/Transformers/bfloat16/1 --levers "$LEVERS" --name "ARC AGI2 Union Sft141"` | `takumuhata/arc-agi2-union-sft141` |
| swap, do not union | not expected for a 2B model; treat as `no union` and re-check the panel downloads | - |
| no union | keep the result of 1.4 (NVARC union, or sft139 alone) | - |

`--name` keeps this build out of `submit_ag2_union/` (the NVARC union directory). Check before pushing:

    python3 -c "import json;d=json.load(open('submit_ag2_union_sft141/kernel-metadata.json'));print(d['id'],d['machine_shape'],d['model_sources'])"
    # takumuhata/arc-agi2-union-sft141 NvidiaL4 [sft139 ..., 'sorokin/qwen3_2b_grids15_sft141/Transformers/bfloat16/1']

## Score scales (more likely to matter here than with NVARC)

A 2B model's NLLs are not on the 4B model's scale. After the first union run that has pass-2 output
(a `--whitelist $PANEL` build of the command above is enough):

    python3 arc-agi-2/dev/swap_eval.py union-scores <that run's inference_outputs> --name sft141

Add `--calibrate offset` (or `ratio`) to the production build only if its VERDICT line says so.

## Choosing between two unions

There is no three-pass kernel: one second model per build. If both union lines say `build`:
1. higher `fallback union` pass@2 in `swap_eval.py union` wins;
2. tie -> sft141 (0.39x the pass-2 time leaves more of the rerun budget to pass 1);
3. the other one becomes the second final selection (section 1.6: most different, not worse by more than one task).

Unverified: sft141 has only been checked with `check_swap_model.py` (DROP-IN OK on the real
config, tokenizer and safetensors header); it has not been run through TTT or decoding on a GPU.
The panel is that test.
