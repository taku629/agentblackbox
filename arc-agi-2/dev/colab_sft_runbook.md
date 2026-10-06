# colab_sft_ag2.ipynb -- per-cell runbook

Companion to `colab_sft_ag2.ipynb` (built by `make_colab_sft_nb.py`). One notebook = one
run = one private Kaggle Model. Everything is resumable from Google Drive; after a
disconnect, Runtime > Run all picks up at the last checkpoint (`--resume` is automatic).

## Per-cell pass lines

| # | cell | pass looks like |
|---|------|-----------------|
| 0 | (markdown) | -- |
| 1 | GPU detect | `preset: t4` (or l4/a100). Fails loud on a CPU runtime -- switch GPU first |
| 2 | deps | `torch 2.x | cuda: True` |
| 3 | auth | `owner: takumuhata | out: /content/drive/MyDrive/arc_sft/<slug>` and a `kaggle models list` table (even empty). Secret missing = `assert cred` |
| 4 | unpack | `10 files under /content/repo` |
| 5 | base + selftests | `check_swap_model` prints DROP-IN OK; `check_sft_data --selftest` and `sft_ag2 --selftest` print `selftest ok` |
| 6 | data | `check_sft_data` summary: every source `format OK`, `0 overlapping the evaluation set`; `--verify` line for synth; `exclude.json` written |
| 7 | train | streams `step/loss` lines; ends with `DROP-IN OK` + `manifest.json written`. A `resuming` line appears only after a disconnect |
| 8 | gate | prints the manifest and `GATE PASS`. If it asserts, STOP -- do not upload (see gate below) |
| 9 | upload | `rc: 0` lines, or `already exists` -> falls back to a new version. UNVERIFIED on Colab -- on failure see "hand over" |
| 10 | verify | `check_swap_model /content/verify /content/base` -> `DROP-IN OK`, then `DONE.` + the model source string |
| 11 | (markdown) | -- |

## If the GPU is a T4

The T4 preset runs fp16 with `--fp16-guard auto` (probes 16 examples; the widening decision
is stored in the checkpoint so a resume is bit-identical).

| symptom | reading | action |
|---------|---------|--------|
| `skipped_updates` grows in the log, loss still finite | fp16 overflow, guard absorbing it | let it run; it self-corrects per step |
| "non-finite losses" abort, `fp16_guard: full` | fp16 unusable for this batch | move to L4/A100 runtime and Run all (checkpoint carries over) |
| s/step > ~45 s at preset t4 | too slow to finish in the free window | same -- L4/A100, or cut `--max-steps` and accept a weaker run |
| OOM in the first 50 steps | preset does not fit | L4/A100 runtime |

The first 50 steps are the hardware test -- watch `nvidia-smi` and the `s/step` figure
before walking away.

## The gate (cell 8)

Both must hold or nothing goes up:

- `drop_in_ok` -- `<out>/merged` is a loadable drop-in for the base (config/tokenizer/
  weights all pass `check_swap_model`)
- `val_improved` -- held-out loss on 50 public train tasks (`--val-from first`) went down.
  This is the check a hurt-the-model adapter would have failed

A failed gate is a datapoint, not a retry: the merged dir stays on Drive; report the
manifest numbers instead of re-running blindly.

## Hand over (ops VM)

Success output: `<owner>/<slug>/Transformers/bf16/<n>`.

    # then on the ops VM (swap_runbook.md step 2): the printed model source goes straight into
    #   make_gen_kernel.py --model-source <owner>/<slug>/Transformers/bf16/<n>
    # or make_union_kernel.py --second <owner>/<slug>/Transformers/bf16/<n>

If cell 9 failed on Colab (`kaggle models` CLI unverified there): the finished model is
already on Drive at `arc_sft/<slug>/merged`. Sync it down and upload from the ops VM
instead, or mount Drive on the ops side -- the merged dir + manifest are all that matters.

## A/B note

`--synth-share 0.0` vs `0.30` (default) generate two notebooks from the same generator;
run the 0.0 one first only if you suspect synthetic tasks hurt val. Thirty percent is a
starting point, not a measured optimum -- the panel decides.
