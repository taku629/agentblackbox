# Colab: SFT of the AGI-2 base checkpoint -> private Kaggle Model

Run this only if `gen_autopsy.py` section 4 says `SFT plausible` or `mixed`
(and, better, after `swap_eval.py compare` shows xcalibur moved the gold NLL).
Nothing below pushes a kernel or submits; the last step uploads a private model.

Status: the script is verified on CPU with a tiny random model
(`sft_ag2.py --selftest`). No preset has been run on a GPU yet -- the first
real run is the hardware test. Memory/time figures in the script are estimates.

## 0. What to upload to the runtime

Build one archive on the ops VM (keeps the repo layout, which the scripts rely on):

    cd <repo>
    tar czf /tmp/sft_bundle.tgz \
      arc-agi-2/dev/sft_ag2.py arc-agi-2/dev/check_sft_data.py arc-agi-2/dev/check_swap_model.py \
      submit_ag2_perfpatch/out/arc_loader.py \
      arc-agi-2/arc-agi_training_challenges.json arc-agi-2/arc-agi_training_solutions.json \
      arc-agi-2/arc-agi_evaluation_challenges.json arc-agi-2/arc-agi_evaluation_solutions.json

Upload `sft_bundle.tgz` through the Files panel (one file; multi-select uploads
deliver only the first). The evaluation files are needed for the contamination
guard only -- they are never trained on.

## 1. Runtime setup (one code cell each; do not type in the Terminal -- it drops characters)

    !mkdir -p /content/repo && tar xzf /content/sft_bundle.tgz -C /content/repo
    !pip -q install "transformers>=4.55,<5" peft safetensors "kaggle>=2"
    !nvidia-smi --query-gpu=name,memory.total --format=csv

Kaggle credentials: the `KAGGLE_CREDENTIALS` Colab secret is stale (cannot see
private models). Use the OAuth pair from the ops VM, pasted in the Terminal so
it does not land in the notebook:

    mkdir -p /root/.kaggle
    echo <access_token> > /root/.kaggle/access_token
    echo <credentials.json base64> | base64 -d > /root/.kaggle/credentials.json

## 2. Base checkpoint + self-test

    !kaggle models instances versions download sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1 -p /content/base --untar
    !cd /content/repo && python arc-agi-2/dev/check_swap_model.py /content/base /content/base
    !cd /content/repo && python arc-agi-2/dev/check_sft_data.py --selftest && python arc-agi-2/dev/sft_ag2.py --selftest

(The public `kagglehub.model_download("sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")`
also works unauthenticated; pass the directory it prints as `--model`.)

## 3. Data check (writes the exclude list the trainer also enforces on its own)

    !cd /content/repo && python arc-agi-2/dev/check_sft_data.py \
        tasks:arc-agi-2/arc-agi_training_challenges.json:arc-agi-2/arc-agi_training_solutions.json \
        --write-exclude /content/exclude.json

Measured on the ops side: 1,000 public training tasks, format OK, 0 overlapping
the evaluation set. `Nabidnur/arc-agi-2-grids` sft128 is these same 1,000 tasks
x 128 augmentations (format OK, 0 overlap) -- it adds no tasks, so it is not
needed. Its 410 synthetic tasks are extra data (optional second `--data`):

    !wget -q -O /content/syn.jsonl https://huggingface.co/datasets/Nabidnur/arc-agi-2-grids/resolve/main/synthetic/curriculum_v0_verified.jsonl

## 4. Train (resumable)

Pick the preset from the GPU you got: `t4` (free tier), `l4`, `a100`.

    !cd /content/repo && python arc-agi-2/dev/sft_ag2.py --model /content/base --out /content/drive/MyDrive/sft_run1 \
        --preset t4 --max-steps 4000 --max-hours 10.5 --exclude /content/exclude.json \
        --data tasks:arc-agi-2/arc-agi_training_challenges.json:arc-agi-2/arc-agi_training_solutions.json

Write `--out` to Drive so the checkpoint survives a recycled runtime. After a
stop (time budget or disconnect) run the SAME command plus `--resume`.

First 50 steps tell you whether the preset fits: watch `nvidia-smi` for the
peak and the `s/step` in the log. fp16 overflow on a T4 is handled by the
trainer's fp16 guard (see colab_sft_runbook.md section 3); the one-notebook
version of this whole procedure is built by make_colab_sft_nb.py.
Keep the first run short on purpose (xcalibur was 63 updates): 4,000 steps at
accum 8 = 500 updates.

## 5. Result

The script prints train/val loss, runs `check_swap_model` on `<out>/merged`
and writes `<out>/manifest.json`. Go on only if BOTH hold:

- `DROP-IN OK`
- `val_improved: true` (held-out public tasks; the AGI-3 adapter that hurt the
  model would have failed this check)

## 6. Upload as a private Kaggle Model (first time)

    !mkdir -p /content/meta && kaggle models init -p /content/meta
    # edit model-metadata.json: ownerSlug=takumuhata, title, slug=<slug>, isPrivate=true
    !kaggle models create -p /content/meta
    !kaggle models instances init -p /content/drive/MyDrive/sft_run1/merged
    # edit model-instance-metadata.json: ownerSlug, modelSlug=<slug>, instanceSlug=bf16, framework=transformers
    !kaggle models instances create -p /content/drive/MyDrive/sft_run1/merged

Later runs: `kaggle models instances versions create takumuhata/<slug>/transformers/bf16 -p <merged dir> -n "<notes>"`.
If `framework=transformers` is rejected, use `Transformers` (the enum spelling
was not verified against the live API).

## 7. Verify the uploaded copy and hand over

    !kaggle models instances versions download takumuhata/<slug>/transformers/bf16/1 -p /content/verify --untar
    !cd /content/repo && python arc-agi-2/dev/check_swap_model.py /content/verify /content/base

Then on the ops VM: `swap_runbook.md` from step 2 with
`--model-source takumuhata/<slug>/Transformers/bf16/1`.
