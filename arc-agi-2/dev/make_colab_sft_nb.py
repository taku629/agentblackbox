"""Build arc-agi-2/colab_sft_ag2.ipynb: one self-contained Colab notebook that fine-tunes the AGI-2 base
checkpoint and uploads the merged bf16 model as a private Kaggle Model.

    python3 make_colab_sft_nb.py [--out ../colab_sft_ag2.ipynb] [--owner takumuhata] [--slug qwen3-4b-grids15-sft-a]
    python3 make_colab_sft_nb.py --selftest

The repo is archived and not pushed, so the notebook carries everything it needs: the six scripts and the
four public ARC-AGI-2 json files are embedded (gzip + base64, ~0.6 MB) and written to /content/repo with
the repo's own layout, each checked against its sha256. Nothing is uploaded by hand except Kaggle credentials.
Run this builder again whenever one of the embedded scripts changes; the notebook prints their hashes.

Cells (all plain Python, no shell magics, so every cell can be compiled and is in --selftest):
  1 config        the only cell to edit: model slug, step budget, data shares
  2 gpu           GPU name/VRAM -> preset (t4 / l4 / a100); refuses to continue without CUDA
  3 deps          transformers<5, peft, safetensors, kaggle, kagglehub
  4 credentials   Colab secret KAGGLE_CREDENTIALS, or /content/credentials.json, or an upload prompt
  5 files         embedded scripts + data -> /content/repo, hashes verified
  6 base          sorokin/qwen3_4b_grids15_sft139 (public) via kagglehub; check_swap_model base vs base
  7 data          gen_synth_ag2 tasks, optional Nabidnur 410, check_sft_data on every source -> exclude.json
  8 train         sft_ag2.py, resumable (--resume is added automatically when a checkpoint exists), output on Drive
  9 gate          manifest must say drop_in_ok and val_improved, otherwise the notebook stops here
 10 upload        private Kaggle Model (create or new version)
 11 verify        download the uploaded copy, check_swap_model against the base, print the model source string
"""
import base64
import gzip
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EMBED = [
    "arc-agi-2/dev/sft_ag2.py", "arc-agi-2/dev/check_sft_data.py", "arc-agi-2/dev/check_swap_model.py",
    "arc-agi-2/dev/gen_synth_ag2.py", "arc-agi-2/dev/dsl_all.py", "submit_ag2_perfpatch/out/arc_loader.py",
    "arc-agi-2/arc-agi_training_challenges.json", "arc-agi-2/arc-agi_training_solutions.json",
    "arc-agi-2/arc-agi_evaluation_challenges.json", "arc-agi-2/arc-agi_evaluation_solutions.json",
]

MD_TOP = """# ARC-AGI-2: SFT of the base checkpoint on Colab -> private Kaggle Model

Fine-tunes `sorokin/qwen3_4b_grids15_sft139` (LoRA, merged at the end) on the 1,000 public training tasks
plus generated synthetic tasks, and uploads the merged bf16 model as a **private** Kaggle Model that the
swap pipeline can mount (`--model-source <owner>/<slug>/Transformers/bf16/<n>`).

**Before running**
1. Runtime -> Change runtime type -> GPU. T4 works (slow, fp16 with an overflow guard); L4 or A100 is better.
2. Kaggle credentials for the account that will own the model: Colab secret `KAGGLE_CREDENTIALS`
   (contents of `credentials.json` or `kaggle.json`), or the file at `/content/credentials.json`.
3. Edit the config cell if needed, then Runtime -> Run all. A recycled runtime: Run all again, training resumes.

It never pushes a kernel and never submits. The 120 evaluation tasks are embedded only for the
contamination guard; any task that shares a pair with them is dropped before training.
"""

CELL_CONFIG = '''# 1. CONFIG -- the only cell to edit
OWNER = %(owner)r                 # Kaggle account that will own the model
MODEL_SLUG = %(slug)r             # new private model; reruns add versions
RUN_NAME = "sft_run1"             # output folder name (on Drive when USE_DRIVE)
USE_DRIVE = True                  # keep checkpoints across recycled runtimes
PRESET = "auto"                   # auto | t4 | l4 | a100
MAX_STEPS = 4000                  # sequences; 8 per update -> 500 updates (xcalibur's whole run was 63)
MAX_HOURS = 10.5                  # stop cleanly before the session limit; Run all again resumes
N_SYNTH = 3000                    # gen_synth_ag2 tasks (0 = none)
SYNTH_SHARE = 0.30                # share of training sequences drawn from them
USE_NABIDNUR_410 = True           # the 410 verified synthetic tasks of Nabidnur/arc-agi-2-grids
NABIDNUR_SHARE = 0.05
FP16_GUARD = "auto"               # T4 only: auto | full | off (see sft_ag2.py)
RUN_SELFTESTS = True              # ~3 min on the Colab CPU; proves the embedded code runs here
UPLOAD = True                     # False = train and gate only
BASE_SOURCE = "sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
assert 0 <= SYNTH_SHARE + NABIDNUR_SHARE < 1 and MODEL_SLUG and OWNER
'''

CELL_GPU = '''# 2. GPU -> preset
import subprocess, sys

def pick_preset(name, gib, wanted="auto"):
    """GPU name + memory -> sft_ag2 preset. bf16 presets need a GPU with native bf16 (not T4/P100/K80/V100)."""
    n = name.lower()
    no_bf16 = any(t in n for t in ("t4", "p100", "k80", "v100", "p4"))
    auto = "t4" if no_bf16 else ("a100" if gib >= 38 else "l4")
    if wanted == "auto":
        return auto
    if wanted in ("l4", "a100") and no_bf16:
        raise SystemExit(f"preset {wanted} needs bf16; {name} has none -- use t4")
    if wanted == "a100" and gib < 38:
        raise SystemExit(f"preset a100 needs ~40 GB; {name} has {gib:.0f} GB -- use l4")
    return wanted

def sh(cmd, check=True, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    r = subprocess.run(cmd, shell=isinstance(cmd, str), **kw)
    if check and r.returncode:
        raise SystemExit(f"command failed with exit code {r.returncode}")
    return r

q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True)
assert q.returncode == 0 and q.stdout.strip(), "no GPU: Runtime -> Change runtime type -> GPU"
GPU_NAME, GPU_MIB = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
PRESET_USED = pick_preset(GPU_NAME, float(GPU_MIB) / 1024, PRESET)
print(f"GPU {GPU_NAME} ({float(GPU_MIB) / 1024:.0f} GiB) -> preset {PRESET_USED}")
'''

CELL_DEPS = '''# 3. dependencies (torch comes with Colab)
sh([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.55,<5", "peft>=0.15", "safetensors", "kaggle>=1.7", "kagglehub"])
import torch, transformers, peft
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
assert torch.cuda.is_available()
'''

CELL_CREDS = '''# 4. Kaggle credentials (never printed)
import json, os, pathlib, stat
kd = pathlib.Path.home() / ".kaggle"
kd.mkdir(exist_ok=True)
raw = None
try:
    from google.colab import userdata
    raw = userdata.get("KAGGLE_CREDENTIALS")
except Exception:
    raw = None
for p in ("/content/credentials.json", "/content/kaggle.json"):
    if not raw and os.path.exists(p):
        raw = open(p).read()
if not raw:
    from google.colab import files
    raw = next(iter(files.upload().values())).decode()
cred = json.loads(raw)
name = "kaggle.json" if "key" in cred else "credentials.json"       # legacy API key vs OAuth credentials
(kd / name).write_text(json.dumps(cred))
os.chmod(kd / name, stat.S_IRUSR | stat.S_IWUSR)
who = cred.get("username") or cred.get("user_name") or "?"
print("kaggle credentials written:", name, "| user:", who)
if who not in ("?", OWNER):
    print(f"WARNING: credentials are for {who!r} but OWNER is {OWNER!r}; the upload cell will use OWNER")
'''

CELL_FILES = '''# 5. embedded scripts + public ARC-AGI-2 json -> /content/repo (repo layout), hashes verified
import base64, gzip, hashlib
FILES = %(files)s
ROOT = pathlib.Path("/content/repo")
for rel, (sha, blob) in FILES.items():
    data = gzip.decompress(base64.b64decode(blob))
    assert hashlib.sha256(data).hexdigest() == sha, f"corrupt embedded file: {rel}"
    (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / rel).write_bytes(data)
    print(f"{sha[:12]}  {len(data):>9,}  {rel}")
DEV = ROOT / "arc-agi-2" / "dev"
TRAIN_SPEC = f"tasks:{ROOT}/arc-agi-2/arc-agi_training_challenges.json:{ROOT}/arc-agi-2/arc-agi_training_solutions.json"
'''

CELL_BASE = '''# 6. base checkpoint (public) + drop-in self-check + selftests of the embedded code
import kagglehub
BASE = kagglehub.model_download(BASE_SOURCE)
print("base:", BASE)
sh([sys.executable, str(DEV / "check_swap_model.py"), BASE, BASE])
if RUN_SELFTESTS:
    sh([sys.executable, str(DEV / "check_sft_data.py"), "--selftest"])
    sh([sys.executable, str(DEV / "gen_synth_ag2.py"), "--selftest"])
    sh([sys.executable, str(DEV / "sft_ag2.py"), "--selftest"])
'''

CELL_DATA = '''# 7. data: public train (held-out val comes from here) + synthetic sources with fixed sampling shares
import urllib.request
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = pathlib.Path("/content/drive/MyDrive") / RUN_NAME
else:
    OUT = pathlib.Path("/content") / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
DATA = [TRAIN_SPEC]                              # first source: the only one validation tasks are taken from
CHECK = [TRAIN_SPEC]
if N_SYNTH:
    synth = OUT / f"synth_{N_SYNTH}.jsonl"       # kept with the run so a resumed session trains on the same tasks
    if not synth.exists():
        sh([sys.executable, str(DEV / "gen_synth_ag2.py"), "--n", str(N_SYNTH), "--out", str(synth), "--seed", "0", "--exclude-train"])
    sh([sys.executable, str(DEV / "gen_synth_ag2.py"), "--verify", str(synth)])
    DATA.append(f"jsonl:{synth}@{SYNTH_SHARE}")
    CHECK.append(f"jsonl:{synth}")
if USE_NABIDNUR_410:
    nab = OUT / "nabidnur_410.jsonl"
    if not nab.exists():
        try:
            urllib.request.urlretrieve("https://huggingface.co/datasets/Nabidnur/arc-agi-2-grids/resolve/main/"
                                       "synthetic/curriculum_v0_verified.jsonl", nab)
        except Exception as exc:
            print("Nabidnur 410 not downloaded, continuing without it:", repr(exc)[:200])
    if nab.exists():
        DATA.append(f"jsonl:{nab}@{NABIDNUR_SHARE}")
        CHECK.append(f"jsonl:{nab}")
EXCLUDE = OUT / "exclude.json"
sh([sys.executable, str(DEV / "check_sft_data.py")] + CHECK + ["--write-exclude", str(EXCLUDE)])
print("training sources:", DATA)
'''

CELL_TRAIN = '''# 8. train (resumable: when a checkpoint exists the same command continues it)
cmd = [sys.executable, str(DEV / "sft_ag2.py"), "--model", BASE, "--out", str(OUT), "--preset", PRESET_USED,
       "--max-steps", str(MAX_STEPS), "--max-hours", str(MAX_HOURS), "--exclude", str(EXCLUDE),
       "--val-from", "first", "--fp16-guard", FP16_GUARD, "--log-every", "25", "--save-every", "200"]
for spec in DATA:
    cmd += ["--data", spec]
if (OUT / "ckpt.pt").exists():
    cmd.append("--resume")
if (OUT / "manifest.json").exists():
    print("manifest.json already exists in", OUT, "-- training is finished; delete it or change RUN_NAME to train again")
else:
    sh(cmd)
'''

CELL_GATE = '''# 9. gate: stop here unless the run finished, is a drop-in and improved held-out loss
mf = OUT / "manifest.json"
if not mf.exists():
    raise SystemExit("training stopped before the last step (time budget or disconnect): Run all again to resume")
M = json.loads(mf.read_text())
print(json.dumps({k: M[k] for k in ("steps", "updates", "train_loss_first", "train_loss_last", "val_loss_before",
                                    "val_loss_after", "val_improved", "non_finite_steps", "skipped_updates",
                                    "fp16_guard", "drop_in_ok") if k in M}, indent=1))
assert M["drop_in_ok"], "merged model is NOT a drop-in: do not upload"
assert M["val_improved"], "held-out loss got worse: do not upload (lower the learning rate or the step count)"
MERGED = OUT / "merged"
'''

CELL_UPLOAD = '''# 10. upload as a PRIVATE Kaggle Model (first run creates it, later runs add a version)
if not UPLOAD:
    raise SystemExit("UPLOAD is False: stopping after the gate")
meta = pathlib.Path("/content/model_meta")
meta.mkdir(exist_ok=True)
(meta / "model-metadata.json").write_text(json.dumps({
    "ownerSlug": OWNER, "title": MODEL_SLUG, "slug": MODEL_SLUG, "subtitle": "", "isPrivate": True,
    "description": "sft139 + LoRA SFT (merged, bf16). Private.", "publishTime": "", "provenanceSources": ""}, indent=1))
notes = f"{RUN_NAME}: {M['updates']} updates, val {M['val_loss_before']:.4f} -> {M['val_loss_after']:.4f}"
(MERGED / "model-instance-metadata.json").write_text(json.dumps({
    "ownerSlug": OWNER, "modelSlug": MODEL_SLUG, "instanceSlug": "bf16", "framework": "transformers",
    "overview": notes, "usage": "drop-in for sorokin/qwen3_4b_grids15_sft139 (16-token vocabulary)",
    "licenseName": "Apache 2.0", "fineTunable": True, "trainingData": [], "modelInstanceType": "Unspecified",
    "baseModelInstanceId": 0, "externalBaseModelUrl": ""}, indent=1))
sh(["kaggle", "models", "create", "-p", str(meta)], check=False)             # "already exists" is fine
first = sh(["kaggle", "models", "instances", "create", "-p", str(MERGED)], check=False)
if first.returncode:
    sh(["kaggle", "models", "instances", "versions", "create", f"{OWNER}/{MODEL_SLUG}/transformers/bf16",
        "-p", str(MERGED), "-n", notes])
'''

CELL_VERIFY = '''# 11. verify the uploaded copy and print what to hand over
import re, shutil, time
ver = pathlib.Path("/content/verify")
shutil.rmtree(ver, ignore_errors=True)
listing = subprocess.run(["kaggle", "models", "instances", "versions", "list", f"{OWNER}/{MODEL_SLUG}/transformers/bf16"],
                         capture_output=True, text=True).stdout
nums = [int(x) for x in re.findall(r"^\\s*(\\d+)\\s", listing, flags=re.M)]
VERSION = max(nums) if nums else 1
for attempt in range(10):                        # a new version takes a few minutes to become downloadable
    r = sh(["kaggle", "models", "instances", "versions", "download",
            f"{OWNER}/{MODEL_SLUG}/transformers/bf16/{VERSION}", "-p", str(ver), "--untar"], check=False)
    if r.returncode == 0 and any(ver.glob("*.safetensors")):
        break
    time.sleep(60)
else:
    raise SystemExit("uploaded version did not become downloadable in 10 minutes; check the model page")
sh([sys.executable, str(DEV / "check_swap_model.py"), str(ver), BASE])
print("\\nHAND OVER:  --model-source", f"{OWNER}/{MODEL_SLUG}/Transformers/bf16/{VERSION}")
print("manifest:", mf)
'''


def embed():
    out = {}
    for rel in EMBED:
        data = open(os.path.join(REPO, rel), "rb").read()
        out[rel] = (hashlib.sha256(data).hexdigest(), base64.b64encode(gzip.compress(data, 9, mtime=0)).decode())
    return out


def cells(owner, slug):
    files = "{\n" + "".join("    %r: (%r,\n        %r),\n" % (k, v[0], v[1]) for k, v in embed().items()) + "}"
    return [("markdown", MD_TOP), ("code", CELL_CONFIG % dict(owner=owner, slug=slug)), ("code", CELL_GPU),
            ("code", CELL_DEPS), ("code", CELL_CREDS), ("code", CELL_FILES % dict(files=files)), ("code", CELL_BASE),
            ("code", CELL_DATA), ("code", CELL_TRAIN), ("code", CELL_GATE), ("code", CELL_UPLOAD), ("code", CELL_VERIFY)]


def build(owner="takumuhata", slug="qwen3-4b-grids15-sft-a"):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                       "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(cells(owner, slug)):
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": "sft%02d" % i}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    return nb


def selftest():
    import subprocess
    import tempfile
    nb = build()
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 11 and json.loads(json.dumps(nb)) == nb
    for i, src in enumerate(code):
        compile(src, "cell%d" % (i + 1), "exec")                       # no shell magics anywhere
        assert not any(ln.lstrip().startswith(("!", "%")) for ln in src.splitlines()), i
    # the config cell runs, and the preset choice is right for the GPUs Colab hands out
    ns = {}
    exec(code[0], ns)
    gpu = code[1].split("q = subprocess.run")[0]
    exec(gpu, ns)
    pick = ns["pick_preset"]
    assert [pick(n, g) for n, g in (("Tesla T4", 15), ("NVIDIA L4", 22.5), ("NVIDIA A100-SXM4-40GB", 40),
                                    ("NVIDIA A100-SXM4-80GB", 80), ("Tesla V100-SXM2-16GB", 16))] == ["t4", "l4", "a100", "a100", "t4"]
    assert pick("NVIDIA L4", 22.5, "t4") == "t4"
    for bad in (("Tesla T4", 15, "l4"), ("NVIDIA L4", 22.5, "a100")):
        try:
            pick(*bad)
            raise AssertionError("accepted %r" % (bad,))
        except SystemExit:
            pass
    # the files cell reproduces every embedded file byte for byte, and the extracted tree runs
    with tempfile.TemporaryDirectory() as tmp:
        src = code[4].replace('pathlib.Path("/content/repo")', "pathlib.Path(%r)" % tmp)
        assert src != code[4]
        ns2 = {"pathlib": __import__("pathlib"), "print": lambda *a, **k: None}
        exec(src, ns2)
        for rel in EMBED:
            assert open(os.path.join(tmp, rel), "rb").read() == open(os.path.join(REPO, rel), "rb").read(), rel
        assert ns2["TRAIN_SPEC"].startswith("tasks:" + tmp)
        dev = os.path.join(tmp, "arc-agi-2", "dev")
        for script, args in (("check_sft_data.py", ["--selftest"]), ("check_swap_model.py", ["--selftest"]),
                             ("gen_synth_ag2.py", ["--n", "40", "--out", os.path.join(tmp, "s.jsonl"), "--seed", "0", "--exclude-train"]),
                             ("gen_synth_ag2.py", ["--verify", os.path.join(tmp, "s.jsonl")])):
            r = subprocess.run([sys.executable, os.path.join(dev, script)] + args, capture_output=True, text=True)
            assert r.returncode == 0, (script, r.stdout[-400:], r.stderr[-400:])
        # the data cell's sources pass the trainer's own loader with the mix the notebook asks for
        sys.path.insert(0, dev)
        try:
            import importlib
            mod = importlib.import_module("sft_ag2")              # module level needs numpy only, not torch
            specs = [ns2["TRAIN_SPEC"], "jsonl:%s@0.3" % os.path.join(tmp, "s.jsonl")]
            tr, va, source = mod.load_tasks(specs, log=lambda *_: None, val_from="first", with_sources=True)
            mix = mod.make_mix(specs, tr, source)
            assert all(source[t] == 0 for t in va) and len(va) == 50 and abs(mix[0][0] - 0.7) < 1e-9
            assert sum(source[t] == 1 for t in tr) == 40 and sum(source[t] == 0 for t in tr) == 950
        finally:
            sys.path.remove(dev)
    # the train command the notebook builds is accepted by the trainer's argument parser
    tr_src = code[7]
    for flag in ("--val-from", "--fp16-guard", "--max-hours", "--exclude", "--preset", "--resume"):
        assert flag in tr_src
    args = mod.parser().parse_args(["--model", "b", "--out", "o", "--preset", "t4", "--max-steps", "4000", "--max-hours", "10.5",
                                    "--exclude", "e", "--val-from", "first", "--fp16-guard", "auto", "--log-every", "25",
                                    "--save-every", "200", "--data", "x", "--data", "jsonl:y@0.3", "--resume"])
    assert args.val_from == "first" and args.fp16_guard == "auto" and args.resume and len(args.data) == 2
    size = len(json.dumps(nb))
    assert size < 2_000_000
    print("selftest ok: 11 code cells compile without shell magics; preset choice T4/L4/A100/V100 correct; the files "
          "cell restores all %d embedded files byte for byte and the restored tree runs check_sft_data, check_swap_model "
          "and gen_synth_ag2; the notebook's data sources load with a 70/30 mix and 50 held-out public tasks; "
          "the train command parses; notebook size %.2f MB" % (len(EMBED), size / 1e6))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--selftest"]:
        selftest()
        sys.exit(0)
    opt = {k: a[a.index(k) + 1] for k in ("--out", "--owner", "--slug") if k in a}
    out = opt.get("--out", os.path.join(os.path.dirname(HERE), "colab_sft_ag2.ipynb"))
    nb = build(opt.get("--owner", "takumuhata"), opt.get("--slug", "qwen3-4b-grids15-sft-a"))
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print("wrote %s (%.2f MB, %d cells). Embedded:" % (out, os.path.getsize(out) / 1e6, len(nb["cells"])))
    for rel, (sha, _) in embed().items():
        print("  %s  %s" % (sha[:16], rel))
