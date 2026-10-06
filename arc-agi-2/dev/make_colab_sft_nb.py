"""Generate the self-contained Colab SFT notebook for the AGI-2 base checkpoint.

    python3 make_colab_sft_nb.py --slug qwen3-4b-grids15-sft-a
        -> writes arc-agi-2/colab_sft_ag2.ipynb (~0.6 MB)
    python3 make_colab_sft_nb.py --slug qwen3-4b-grids15-sft-a --synth-share 0.0 --out colab_sft_ag2_nosynth.ipynb
    python3 make_colab_sft_nb.py --selftest

The repo cannot be pushed to a place Colab can reach, so the notebook embeds the 6 scripts
and the 4 public ARC-AGI-2 JSONs as one zlib+base64 tarball: the only manual upload is the
Kaggle credential (Colab secret `KAGGLE_CREDENTIALS` = contents of ~/.kaggle/credentials.json).
"Run all" goes: GPU check -> deps -> auth -> unpack -> base download -> data build +
contamination check -> train (auto --resume from the Drive checkpoint) -> gate -> upload as
a private Kaggle Model -> re-download + check_swap_model.

Default data mix (the '@share' weights added in part 14a): public train 65% /
gen_synth_ag2 3,000 tasks 30% / Nabidnur 410 synthetic tasks 5%; val = 50 held-out
public-train tasks (--val-from first). 30% is a starting point, not a measured optimum --
settle it with a panel A/B of --synth-share 0.0 vs 0.30.

Cells 9-10 (kaggle CLI upload) are unverified on Colab; on failure the merged model stays
on Drive (<OUT>/merged) and the ops VM can upload it instead. Details per cell:
colab_sft_runbook.md.
"""
import argparse
import base64
import io
import json
import os
import sys
import tarfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# embedded into the notebook, repo-relative layout kept (check_sft_data finds the eval JSONs
# via BASE and arc_loader via PROD, so the paths below are load-bearing, not cosmetic)
FILES = [
    "arc-agi-2/dev/sft_ag2.py",
    "arc-agi-2/dev/check_sft_data.py",
    "arc-agi-2/dev/check_swap_model.py",
    "arc-agi-2/dev/gen_synth_ag2.py",
    "arc-agi-2/dev/dsl_all.py",
    "submit_ag2_perfpatch/out/arc_loader.py",
    "arc-agi-2/arc-agi_training_challenges.json",
    "arc-agi-2/arc-agi_training_solutions.json",
    "arc-agi-2/arc-agi_evaluation_challenges.json",
    "arc-agi-2/arc-agi_evaluation_solutions.json",
]

BASE_MODEL = "sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
NABIDNUR_URL = ("https://huggingface.co/datasets/Nabidnur/arc-agi-2-grids/resolve/main/"
                "synthetic/curriculum_v0_verified.jsonl")
DATA_TRAIN = "tasks:arc-agi-2/arc-agi_training_challenges.json:arc-agi-2/arc-agi_training_solutions.json"
SYNTH_N = 3000
TRAIN_SHARE, SYNTH_SHARE, NAB_SHARE = 0.65, 0.30, 0.05          # synth share is the knob
MAX_STEPS, MAX_HOURS = 4000, 10.5                               # ~500 updates at accum 8


def payload():
    """The embedded files as one zlib-compressed tar -> base64 string."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for rel in FILES:
            tf.add(os.path.join(REPO, rel), arcname=rel)
    return base64.b64encode(zlib.compress(buf.getvalue(), 9)).decode()


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


def build(slug, synth_share=SYNTH_SHARE):
    """-> notebook dict. slug is the Kaggle model slug AND the Drive run directory."""
    assert slug and "/" not in slug and slug.replace("-", "").replace("_", "").isalnum(), slug
    assert 0.0 <= synth_share <= 0.9
    train_share = TRAIN_SHARE + (SYNTH_SHARE - synth_share)     # freed share goes to real tasks
    use_synth = synth_share > 0

    b64 = payload()
    blob = "".join(f'    "{b64[i:i + 100]}"\n' for i in range(0, len(b64), 100))

    # data specs with the '@share' sampling weights (sft_ag2.split_spec / make_mix)
    data_specs = [repr(DATA_TRAIN + "@" + str(train_share))]
    if use_synth:
        data_specs.append('"jsonl:" + SYNTH + "@' + str(synth_share) + '"')
    data_specs.append('"jsonl:" + NAB + "@' + str(NAB_SHARE) + '"')
    data_block = ("SYNTH = OUT + \"/synth.jsonl\"\n"
                  "NAB = OUT + \"/nabidnur.jsonl\"\n"
                  "DATA = [" + ", ".join(data_specs) + "]\n")

    synth_lines = ""
    if use_synth:
        synth_lines = (
            f"if not os.path.exists(SYNTH):\n"
            f"    subprocess.run(f\"cd /content/repo && python arc-agi-2/dev/gen_synth_ag2.py --n {SYNTH_N} \"\n"
            f"                   f\"--out {{SYNTH}} --seed 0\", shell=True, check=True)\n"
            f"subprocess.run(\"cd /content/repo && python arc-agi-2/dev/gen_synth_ag2.py --verify \" + SYNTH,\n"
            f"               shell=True, check=True)\n")

    cells = [
        cell("markdown", f"""# ARC-AGI-2 SFT in Colab -> private Kaggle Model `{slug}`

One notebook = one training run = one private Kaggle Model.

1. Runtime > Change runtime type > GPU (T4 free tier works; L4/A100 finish sooner).
2. Add a Colab secret named `KAGGLE_CREDENTIALS` holding the contents of
   `~/.kaggle/credentials.json` (the OAuth pair, not the legacy key).
3. Runtime > Run all.

Everything is resumable: the checkpoint and generated data live on Google Drive, so after a
disconnect `Run all` picks up at the last saved checkpoint (`--resume`). Do not edit cells
while a run is in flight. Details and per-cell pass lines: `arc-agi-2/dev/colab_sft_runbook.md`.
"""),
        cell("code", """import subprocess
r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                   capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())
name = r.stdout.lower()
PRESET = "a100" if "a100" in name else "l4" if "l4" in name else "t4" if "t4" in name else None
assert PRESET, "no GPU on this runtime -- Runtime > Change runtime type > GPU"
print("preset:", PRESET)
"""),
        cell("code", """import subprocess
subprocess.run("pip -q install 'transformers>=4.55,<5' 'peft>=0.13' safetensors 'kaggle>=2'",
               shell=True, check=True)
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
assert torch.cuda.is_available()
"""),
        cell("code", f"""import json, os, subprocess
from google.colab import userdata, drive

os.makedirs("/root/.kaggle", exist_ok=True)
cred = userdata.get("KAGGLE_CREDENTIALS")
assert cred, "add the Colab secret KAGGLE_CREDENTIALS = contents of ~/.kaggle/credentials.json"
open("/root/.kaggle/credentials.json", "w").write(cred.strip())
os.chmod("/root/.kaggle/credentials.json", 0o600)
OWNER = json.loads(cred).get("username") or "takumuhata"
RUN_NAME = "{slug}"
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/arc_sft/" + RUN_NAME
os.makedirs(OUT, exist_ok=True)
r = subprocess.run(["kaggle", "models", "list", "-m"], capture_output=True, text=True)
print(r.stdout[:400] or r.stderr[:400])
print("owner:", OWNER, "| out:", OUT)
"""),
        cell("code", """# unpack the embedded repo subset (6 scripts + 4 public ARC-AGI-2 JSONs)
import base64, glob, io, tarfile, zlib
BLOB = (
""" + blob + """)
raw = zlib.decompress(base64.b64decode(BLOB))
with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
    tf.extractall("/content/repo", filter="data")
print(len(glob.glob("/content/repo/**/*.*", recursive=True)), "files under /content/repo")
"""),
        cell("code", f"""# base checkpoint + environment self-checks (~3 min CPU for the trainer selftest)
import os, subprocess
if not os.path.exists("/content/base/config.json"):
    subprocess.run("kaggle models instances versions download "
                   "{BASE_MODEL} -p /content/base --untar", shell=True, check=True)
subprocess.run("cd /content/repo && python arc-agi-2/dev/check_swap_model.py /content/base /content/base",
               shell=True, check=True)
subprocess.run("cd /content/repo && python arc-agi-2/dev/check_sft_data.py --selftest", shell=True, check=True)
subprocess.run("cd /content/repo && python arc-agi-2/dev/sft_ag2.py --selftest", shell=True, check=True)
"""),
        cell("code", f"""# data: synthetic tasks + the Nabidnur 410, then the contamination/format check.
# Files live on Drive: regenerated deterministically (seed 0) only if missing.
import os, subprocess
{data_block}if not os.path.exists(NAB):
    subprocess.run("wget -q -O " + NAB + " {NABIDNUR_URL}", shell=True, check=True)
{synth_lines}subprocess.run("cd /content/repo && python arc-agi-2/dev/check_sft_data.py " +
               " ".join(s.split("@")[0] for s in DATA) + " --write-exclude " + OUT + "/exclude.json",
               shell=True, check=True)
"""),
        cell("code", f"""# train (auto-resumes from the Drive checkpoint after a disconnect)
import os, subprocess
cmd = ("cd /content/repo && python arc-agi-2/dev/sft_ag2.py --model /content/base --out " + OUT +
       " --preset " + PRESET + " --max-steps {MAX_STEPS} --max-hours {MAX_HOURS}"
       " --exclude " + OUT + "/exclude.json --val-from first" + "".join(" --data " + s for s in DATA))
if os.path.exists(OUT + "/ckpt.pt"):
    cmd += " --resume"
    print("resuming from the Drive checkpoint")
print(cmd)
subprocess.run(cmd, shell=True, check=True)
"""),
        cell("code", """# gate: nothing leaves this notebook unless the merged model is a drop-in AND val improved
import json
m = json.load(open(OUT + "/manifest.json"))
print(json.dumps({{k: m[k] for k in ("steps", "updates", "train_loss_first", "train_loss_last",
                                    "val_loss_before", "val_loss_after", "val_improved",
                                    "drop_in_ok", "fp16_guard")}}, indent=1))
assert m["drop_in_ok"], "merged dir is not a drop-in for the base -- do not upload"
assert m["val_improved"], "held-out val loss got worse -- do not upload (a panel would waste a run)"
print("GATE PASS -- uploading as a private Kaggle Model")
"""),
        cell("code", """# upload: model resource, then the merged dir as a transformers/bf16 instance.
# UNVERIFIED on Colab: if this fails, merged weights stay on Drive -- upload from the ops VM
# instead (colab_sft_runbook.md, "hand over").
import json, os, subprocess
MERGED = OUT + "/merged"
meta_dir = "/content/meta_" + RUN_NAME
os.makedirs(meta_dir, exist_ok=True)
subprocess.run("kaggle models init -p " + meta_dir, shell=True, check=True)
mp = meta_dir + "/model-metadata.json"
meta = json.load(open(mp))
meta.update(ownerSlug=OWNER, title=RUN_NAME, slug=RUN_NAME, isPrivate=True)
json.dump(meta, open(mp, "w"))
r = subprocess.run("kaggle models create -p " + meta_dir, shell=True)   # 'already exists' is fine
print("model create rc:", r.returncode)
subprocess.run("kaggle models instances init -p " + MERGED, shell=True, check=True)
ip = MERGED + "/model-instance-metadata.json"
im = json.load(open(ip))
im.update(ownerSlug=OWNER, modelSlug=RUN_NAME, instanceSlug="bf16", framework="transformers")
json.dump(im, open(ip, "w"))
r = subprocess.run("kaggle models instances create -p " + MERGED, shell=True)
if r.returncode:     # instance exists -> a new version
    subprocess.run("kaggle models instances versions create " + OWNER + "/" + RUN_NAME +
                   "/transformers/bf16 -p " + MERGED + " -n update", shell=True, check=True)
"""),
        cell("code", """# verify the uploaded copy, then hand over
import subprocess
subprocess.run("kaggle models instances versions download " + OWNER + "/" + RUN_NAME +
               "/transformers/bf16/1 -p /content/verify --untar", shell=True, check=True)
subprocess.run("cd /content/repo && python arc-agi-2/dev/check_swap_model.py /content/verify /content/base",
               shell=True, check=True)
print("DONE. On the ops VM: swap_runbook.md step 2 with "
      "--model-source " + OWNER + "/" + RUN_NAME + "/Transformers/bf16/1")
"""),
        cell("markdown", """Done when the last cell prints DONE. The model source string printed there
goes straight into `make_gen_kernel.py --model-source` (or `make_union_kernel.py --second`)
for the panel comparison -- see `arc-agi-2/dev/swap_runbook.md`.
"""),
    ]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"},
                         "accelerator": "GPU",
                         "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}


def write(nb, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return path


def selftest():
    import tempfile
    for share in (0.3, 0.0):
        nb = build("qwen3-4b-grids15-sft-a", synth_share=share)
        assert nb["nbformat"] == 4 and len(nb["cells"]) == 12
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                compile("".join(c["source"]), "<cell>", "exec")
        src = "".join("".join(c["source"]) for c in nb["cells"])
        assert "--val-from first" in src and "--resume" in src and "userdata.get" in src
        assert "drop_in_ok" in src and "val_improved" in src and "0.05" in src
        if share > 0:
            assert "@0.3" in src and f"gen_synth_ag2.py --n {SYNTH_N}" in src
            assert '"jsonl:" + SYNTH + "@"' not in src  # shares are literals, not expressions
        else:
            assert "gen_synth_ag2.py --verify" not in src and "SYNTH" not in src.split("DATA =")[1].split("]")[0]
        # the payload extracts to byte-identical copies
        blob_cell = next("".join(c["source"]) for c in nb["cells"] if "BLOB = (" in "".join(c["source"]))
        seg = blob_cell.split("BLOB = (", 1)[1]
        blob = eval("(" + seg[:seg.index("\n)")] + ")")
        raw = zlib.decompress(base64.b64decode(blob))
        with tempfile.TemporaryDirectory() as d:
            with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
                tf.extractall(d, filter="data")
            for rel in FILES:
                assert open(os.path.join(d, rel), "rb").read() == open(os.path.join(REPO, rel), "rb").read(), rel
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as f:
            p = f.name
        write(nb, p)
        size = os.path.getsize(p)
        os.unlink(p)
        assert size < 1_500_000, size
        print(f"share={share}: notebook {size / 1e6:.2f} MB, {len(FILES)} embedded files byte-identical")
    print("selftest ok: valid nbformat; all code cells compile; mix @ shares with --val-from first; "
          "--synth-share 0.0 drops the synthetic spec and its generator call; gate needs drop_in_ok AND "
          "val_improved; resume watches the Drive ckpt; upload + verify cells carry the slug")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="qwen3-4b-grids15-sft-a")
    ap.add_argument("--synth-share", type=float, default=SYNTH_SHARE, dest="synth_share")
    ap.add_argument("--out", default=os.path.join(REPO, "arc-agi-2", "colab_sft_ag2.ipynb"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    nb = build(a.slug, a.synth_share)
    print("wrote", write(nb, a.out), f"({os.path.getsize(a.out) / 1e6:.2f} MB)",
          "slug:", a.slug, "| synth share:", a.synth_share, "(not uploaded)")
