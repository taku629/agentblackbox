# Colab SFT runbook (AGI-2): from nothing to a private Kaggle Model that passes check_swap_model

Start it now; it does not depend on Saturday's probe_gold numbers. Nothing here pushes a kernel or submits.
Status: every step that can run without a GPU has been run (see "Verified / not verified" at the end).
The first Colab session IS the hardware test.

## 1. Build the notebook (ops VM, 5 s)

    python3 arc-agi-2/dev/make_colab_sft_nb.py --selftest
    python3 arc-agi-2/dev/make_colab_sft_nb.py --slug qwen3-4b-grids15-sft-a
    #   -> arc-agi-2/colab_sft_ag2.ipynb (0.6 MB): scripts + the four public ARC-AGI-2 json files are embedded

Rebuild whenever sft_ag2.py / gen_synth_ag2.py / check_*.py change; the notebook prints the hash of each embedded file.

## 2. Run it (Colab, account programmerengineer629)

1. Upload `colab_sft_ag2.ipynb`, Runtime -> Change runtime type -> GPU (T4 is enough to start).
2. Kaggle credentials: secret `KAGGLE_CREDENTIALS` = contents of the ops VM's `~/.kaggle/credentials.json`
   (or `kaggle.json`), notebook access ON. Without the secret the notebook asks for the file.
3. Runtime -> Run all. A disconnect or the time budget: Run all again -- cell 8 adds `--resume` by itself
   (checkpoint, generated synthetic tasks and exclude list live in `MyDrive/<RUN_NAME>/`).

What each cell must print:

| cell | pass line | if not |
|---|---|---|
| 2 gpu | `GPU ... -> preset t4` (or l4 / a100) | no GPU attached |
| 5 files | 10 lines `sha  size  path` | notebook file damaged: rebuild |
| 6 base | `DROP-IN OK` (base vs itself) and three `selftest ok` | stop: the environment differs from the one tested |
| 7 data | `--verify` ok; check_sft_data lists every source with `0` tasks overlapping evaluation | stop: do not train |
| 8 train | `fp16 guard: ...` (T4), `val loss before training`, then `step N/4000 loss ... s/step` | see section 3 |
| 9 gate | `"drop_in_ok": true` and `"val_improved": true` | do not upload; see section 5 |
| 11 verify | `DROP-IN OK` on the DOWNLOADED copy and `HAND OVER: --model-source ...` | the upload is incomplete |

## 3. T4 and fp16 (the overflow question)

A T4 has no bf16, so the frozen base runs in fp16. Three different things can go wrong, and only the
first two are optimiser problems:

| risk | handled by |
|---|---|
| gradient overflow in the backward pass | GradScaler (loss scaling; a bad update is skipped and the scale lowered) |
| a bad update from one outlier sequence | gradient clipping at 1.0; lr 2e-5 with 3% warm-up; 8-sequence accumulation |
| rounding in the weights being trained | LoRA weights are fp32 ("fp32 master weights"); the merged result is saved in bf16 |
| an ACTIVATION above 65504 in the forward pass -> inf -> NaN loss | nothing above can fix this; the fp16 guard does |

The guard (`--fp16-guard auto`, default): before the first step the first 16 training sequences are
forwarded and the peak |activation| of every decoder layer is measured. One of two lines appears:

    fp16 guard: all layers stay below 32752 (peak ...) -- pure fp16
    fp16 guard: layer K reaches ... (fp16 max 65504) -> layers K.. , norm and lm_head now compute in fp32 (N layers)

In the second case those layers do their arithmetic in fp32 while the weights stay fp16 (no extra weight
memory). The residual stream is promoted once at layer K and never cast back, so nothing after K can
overflow. A non-finite loss later in training widens the fp32 region instead of being skipped, and the
decision is saved in the checkpoint so a resumed run is identical.

Cost: fp32 layers do not use the T4's fp16 tensor cores. If the guard reports most layers in fp32, expect
several times the s/step of the pure-fp16 case -- at that point an L4 runtime (bf16, no guard needed) is
the better use of the hours. Decide from the `s/step` in the first log line:

| first log line | do |
|---|---|
| pure fp16, <= 12 s/step | keep going: 4,000 sequences fit one ~11 h session |
| guard widened, <= 25 s/step | keep going with `MAX_STEPS = 2000` (250 updates, still 4x xcalibur) |
| slower than that, or `out of memory` | switch the runtime to L4 / A100 and Run all (new `RUN_NAME`) |

In the manifest afterwards: `non_finite_steps` should be 0 and `skipped_updates` a handful at most.
More than ~5% skipped updates means the run is not trustworthy even if the losses look fine.

## 4. Data mix (what the defaults are and why)

| source | tasks | share of training sequences | why |
|---|---|---|---|
| public ARC-AGI-2 training set (embedded) | 1,000 (950 train + 50 held out) | 65% | the real distribution; fresh rotation/reflection/colour augmentation every time a task is drawn |
| `gen_synth_ag2.py --n 3000 --exclude-train` | 3,000 | 30% | the only source of NEW tasks we can make; 26 families + two-step programs |
| Nabidnur `curriculum_v0_verified.jsonl` | 410 | 5% | the only extra tasks in that repo |

- The 128k file is not downloaded: it is the same 1,000 public tasks x 128 stored augmentations
  (measured), and the trainer augments on the fly, so it would add bytes and no task.
- Shares are sampling probabilities (`--data jsonl:file@0.30`), not row counts: 3,000 synthetic tasks
  would otherwise be 3/4 of every batch and pull the model toward 26 simple families.
- Validation = 50 public tasks held out by task id (`--val-from first`), never synthetic ones:
  a val loss on generated tasks would only say the model learned the generator.
- The 120 evaluation tasks are embedded for the contamination guard only. Any task sharing a pair with
  them (up to rotation, reflection, recolouring) is dropped by cell 7 and again inside the trainer.
- 30% is a starting point, not a measured optimum. The comparison that decides it is two runs
  (`SYNTH_SHARE = 0.0` vs `0.30`, same steps) judged on the panel, not on val loss.

## 5. Gate and hand-over

Upload only when the manifest says `drop_in_ok` and `val_improved`. If val got worse: halve
`MAX_STEPS` or set `--lr 1e-5` (edit cell 8's command) under a new `RUN_NAME`; do not ship it "to see".

After cell 11 printed `HAND OVER: --model-source takumuhata/<slug>/Transformers/bf16/<n>`:

    python3 arc-agi-2/dev/make_gen_kernel.py --probe-gold --whitelist $PANEL --name "ARC AGI2 Sft A Panel" \
        --model-source takumuhata/<slug>/Transformers/bf16/<n>
    # then swap_runbook.md step 4 (compare against the sft139 panel) and, for a second model, swap_eval.py union

A better val loss is necessary, not sufficient: the panel verdict decides (the AGI-3 adapter looked fine
on its training loss and cut the score by 3x).

## Verified / not verified

Verified on CPU (tiny random model with the kernel's 16-token vocabulary, real data path):
- `sft_ag2.py --selftest`: fp32 and fp16 paths both learn (loss 2.79 -> 2.66), interrupted + resumed runs are
  bit-identical, the merged directory passes check_swap_model with only LoRA-target weights changed.
- fp16 guard: a model whose second layer overflows fp16 is detected by the probe, computed in fp32 from that
  layer on, and its losses match an fp32 reference within 2%; with the guard off the same model gives NaN.
- source shares: a 300-task source at `@0.25` receives 19-31% of 600 draws; validation tasks come from the first source only.
- `make_colab_sft_nb.py --selftest`: all 11 code cells compile, the embedded files are restored byte for byte
  and run, the notebook's train command is accepted by the trainer.

Not verified (needs the first Colab session):
- memory and s/step of every preset on a real GPU; whether the real 3.6B checkpoint needs the guard at all
- gradient checkpointing together with the fp32 layers on CUDA
- the Kaggle CLI calls of cells 10-11 on the Colab runtime (earlier notebooks hit CLI problems there; the
  model is on Drive in `<RUN_NAME>/merged`, so a failed upload loses nothing -- upload from the ops VM instead)
