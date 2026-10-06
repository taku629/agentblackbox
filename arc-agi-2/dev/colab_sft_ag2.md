# Colab: SFT of the AGI-2 base checkpoint -> private Kaggle Model

Run this only if `gen_autopsy.py` section 4 says `SFT plausible` or `mixed`
(and, better, after `swap_eval.py compare` shows the swap moved the gold NLL).
Nothing below pushes a kernel or submits; the last cell uploads a private model.

Status: the trainer is verified on CPU with a tiny random model
(`sft_ag2.py --selftest`) and the notebook generator's `--selftest` checks that every
embedded file lands byte-identical. No preset has been run on a GPU yet -- the first real
run is the hardware test. Memory/time figures are estimates.

## 0. Build the notebook (ops VM, ~10 s)

    python3 arc-agi-2/dev/make_colab_sft_nb.py --slug qwen3-4b-grids15-sft-a
    # -> arc-agi-2/colab_sft_ag2.ipynb (~0.6 MB)

The notebook embeds the 6 scripts and the 4 public ARC-AGI-2 JSONs as a zlib+base64 blob,
so the ONLY thing ever uploaded by hand is the Kaggle credential. `--slug` becomes both the
Kaggle model name and the Drive run directory. `--synth-share 0.0` emits the no-synthetic
control arm instead of the default 0.30 mix.

## 1. Colab setup (the only manual steps)

1. colab.research.google.com -> File > Upload notebook -> `colab_sft_ag2.ipynb`
2. Runtime > Change runtime type -> GPU (T4 free tier works; L4/A100 finish sooner)
3. Key icon (secrets) -> add `KAGGLE_CREDENTIALS` = the contents of
   `~/.kaggle/credentials.json` from the ops VM (the OAuth pair, not the legacy key)
4. Runtime > Run all

Checkpoint + generated data land on Google Drive (`arc_sft/<slug>/`), so a recycled
runtime costs nothing: Run all again and the train cell resumes from `ckpt.pt`.

## 2. What it does (all automatic)

GPU detect -> pip deps -> Kaggle auth -> unpack embedded files to /content/repo ->
download the base checkpoint -> build data (gen_synth 3,000 + Nabidnur 410) and run the
contamination/format check -> train with the preset for your GPU -> gate -> upload as a
private Kaggle Model -> re-download it and `check_swap_model` the uploaded copy.

Data mix (the `@share` sampling weights from part 14a): public train 65% /
gen_synth 3,000 tasks 30% / Nabidnur 410 tasks 5%; val = 50 public-train tasks held out
first (`--val-from first`). Thirty percent is a starting point, not a measured optimum --
settle it with a 0.0-vs-0.30 panel A/B.

Per-cell "this looks right" lines, the T4 decision table, the gate rules and the hand-over:
`colab_sft_runbook.md`.

## 3. Gate and hand-over

The gate cell stops the notebook unless BOTH hold -- `drop_in_ok` (merged dir is a real
drop-in for the base) and `val_improved` (held-out loss went down). If it fails, keep the
manifest numbers and do not upload.

On success the last cell prints `DONE.` plus the model source string, which feeds
`swap_runbook.md` step 2 (`--model-source` for `make_gen_kernel.py`, or `--second` for
`make_union_kernel.py`).

If the `kaggle models` upload cell fails on Colab (that CLI path is unverified there), the
merged model is already safe on Drive at `arc_sft/<slug>/merged` -- pull it down and upload
from the ops VM instead.
