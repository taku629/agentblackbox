"""Build arc-agi-2/lightning_sft_ag2.ipynb: the Lightning AI Studio variant of colab_sft_ag2.ipynb.

    python3 make_lightning_sft_nb.py [--out ../lightning_sft_ag2.ipynb] [--owner takumuhata] [--slug qwen3-4b-grids15-sft-a]
    python3 make_lightning_sft_nb.py --selftest

Same pipeline as make_colab_sft_nb.py (same embedded files, same trainer, same gate) with the three
Colab-specific pieces swapped for Lightning equivalents:

- credentials:  Lightning Studio secret KAGGLE_CREDENTIALS (env var) or a credentials.json you
                drop in the Studio file browser; ~/.kaggle/ is written either way
- storage:      no Drive mount -- a Studio's own filesystem persists across restarts, so WORK is
                simply the notebook's directory (everything incl. the base checkpoint cache is
                pinned under WORK so an ephemeral HOME costs nothing)
- name:         Lightning GPU names are identical (T4/L4/A100); the shared pick_preset cell applies

Why bother: the free tier ships ~80 GPU-hours (T4 ~75 h, L4 ~31 h, A100-40 ~10 h) + a monthly
15-credit top-up, and a Studio keeps its disk -- the Colab runtime recycle dance is unnecessary.
Students get 80% off paid features with an academic email.

Note the shared cells are imported from make_colab_sft_nb -- regenerate BOTH notebooks when any of
sft_ag2.py / gen_synth_ag2.py / check_*.py changes.
"""
import json
import os
import pathlib
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_colab_sft_nb import EMBED, embed, CELL_GPU, CELL_DEPS, CELL_BASE, CELL_TRAIN, CELL_GATE  # noqa: E402

MD_TOP = """# ARC-AGI-2: SFT of the base checkpoint on Lightning AI Studio -> private Kaggle Model

Lightning AI variant of `colab_sft_ag2.ipynb` -- same embedded files, same trainer, same gate.
Fine-tunes `sorokin/qwen3_4b_grids15_sft139` (LoRA, merged at the end) on the 1,000 public training
tasks plus generated synthetic tasks, and uploads the merged bf16 model as a **private** Kaggle Model
that the swap pipeline can mount (`--model-source <owner>/<slug>/Transformers/bf16/<n>`).

**Before running**
1. In the Studio: pick a GPU machine (T4 works, slow, fp16 with an overflow guard; L4 or A100 is better).
2. Kaggle credentials for the account that will own the model: Studio secret `KAGGLE_CREDENTIALS`
   (contents of `credentials.json` or `kaggle.json`), or drop the file in the file browser next to
   this notebook.
3. Edit the config cell if needed, then run all cells. A restart costs nothing: the Studio disk
   persists, and the train cell resumes from `ckpt.pt` on its own.

It never pushes a kernel and never submits. The 120 evaluation tasks are embedded only for the
contamination guard; any task that shares a pair with them is dropped before training.
"""

CELL_CONFIG = '''# 1. CONFIG -- the only cell to edit
import os, pathlib
OWNER = %(owner)r                 # Kaggle account that will own the model
MODEL_SLUG = %(slug)r             # new private model; reruns add versions
RUN_NAME = "sft_run1"             # output folder name
WORK = pathlib.Path.cwd()         # the Studio filesystem persists across restarts; HOME may not
PRESET = "auto"                   # auto | t4 | l4 | a100
MAX_STEPS = 4000                  # sequences; 8 per update -> 500 updates (xcalibur's whole run was 63)
MAX_HOURS = 10.5                  # stop cleanly and resume on a later run
N_SYNTH = 3000                    # gen_synth_ag2 tasks (0 = none)
SYNTH_SHARE = 0.30                # share of training sequences drawn from them
USE_NABIDNUR_410 = True           # the 410 verified synthetic tasks of Nabidnur/arc-agi-2-grids
NABIDNUR_SHARE = 0.05
FP16_GUARD = "auto"               # T4 only: auto | full | off (see sft_ag2.py)
RUN_SELFTESTS = True              # ~3 min CPU; proves the embedded code runs here
UPLOAD = True                     # False = train and gate only
BASE_SOURCE = "sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
os.environ.setdefault("KAGGLEHUB_CACHE", str(WORK / "cache"))      # keep the 8 GB base on persistent disk
assert 0 <= SYNTH_SHARE + NABIDNUR_SHARE < 1 and MODEL_SLUG and OWNER
print("workspace:", WORK)
'''

CELL_CREDS = '''# 4. Kaggle credentials (never printed)
import json, os, pathlib, stat
kd = pathlib.Path.home() / ".kaggle"
kd.mkdir(exist_ok=True)
raw = os.environ.get("KAGGLE_CREDENTIALS")          # Studio secret (env var)
for p in (WORK / "credentials.json", WORK / "kaggle.json",
          pathlib.Path.home() / "credentials.json", pathlib.Path.home() / "kaggle.json"):
    if not raw and p.exists():
        raw = p.read_text()
if not raw:
    raise SystemExit("no credentials: add Studio secret KAGGLE_CREDENTIALS (env var) "
                     "or drop credentials.json next to this notebook")
cred = json.loads(raw)
name = "kaggle.json" if "key" in cred else "credentials.json"       # legacy API key vs OAuth credentials
(kd / name).write_text(json.dumps(cred))
os.chmod(kd / name, stat.S_IRUSR | stat.S_IWUSR)
who = cred.get("username") or cred.get("user_name") or "?"
print("kaggle credentials written:", name, "| user:", who)
if who not in ("?", OWNER):
    print(f"WARNING: credentials are for {who!r} but OWNER is {OWNER!r}; the upload cell will use OWNER")
'''

CELL_FILES = '''# 5. embedded scripts + public ARC-AGI-2 json -> WORK/repo (repo layout), hashes verified
import base64, gzip, hashlib
FILES = %(files)s
ROOT = WORK / "repo"
for rel, (sha, blob) in FILES.items():
    data = gzip.decompress(base64.b64decode(blob))
    assert hashlib.sha256(data).hexdigest() == sha, f"corrupt embedded file: {rel}"
    (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / rel).write_bytes(data)
    print(f"{sha[:12]}  {len(data):>9,}  {rel}")
DEV = ROOT / "arc-agi-2" / "dev"
TRAIN_SPEC = f"tasks:{ROOT}/arc-agi-2/arc-agi_training_challenges.json:{ROOT}/arc-agi-2/arc-agi_training_solutions.json"
'''

CELL_DATA = '''# 7. data: public train (held-out val comes from here) + synthetic sources with fixed sampling shares
import urllib.request
OUT = WORK / RUN_NAME                              # persistent Studio disk; survives restarts
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

CELL_UPLOAD = '''# 10. upload as a PRIVATE Kaggle Model (first run creates it, later runs add a version)
if not UPLOAD:
    raise SystemExit("UPLOAD is False: stopping after the gate")
meta = WORK / "model_meta"
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
ver = WORK / "verify"
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


def cells(owner, slug):
    files = "{\n" + "".join("    %r: (%r,\n        %r),\n" % (k, v[0], v[1]) for k, v in embed().items()) + "}"
    return [("markdown", MD_TOP), ("code", CELL_CONFIG % dict(owner=owner, slug=slug)), ("code", CELL_GPU),
            ("code", CELL_DEPS), ("code", CELL_CREDS), ("code", CELL_FILES % dict(files=files)), ("code", CELL_BASE),
            ("code", CELL_DATA), ("code", CELL_TRAIN), ("code", CELL_GATE), ("code", CELL_UPLOAD), ("code", CELL_VERIFY)]


def build(owner="takumuhata", slug="qwen3-4b-grids15-sft-a"):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
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
    import unittest.mock as mock
    ns_c_pathlib = pathlib
    nb = build()
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 11 and json.loads(json.dumps(nb)) == nb
    for i, src in enumerate(code):
        compile(src, "cell%d" % (i + 1), "exec")
        assert not any(ln.lstrip().startswith(("!", "%")) for ln in src.splitlines()), i
    # config runs standalone; GPU pick is shared with the Colab notebook verbatim
    ns = {}
    exec(code[0], ns)
    assert ns["WORK"].name == os.path.basename(os.getcwd())
    gpu = code[1].split("q = subprocess.run")[0]
    exec(gpu, ns)
    pick = ns["pick_preset"]
    assert [pick(n, g) for n, g in (("Tesla T4", 15), ("NVIDIA L4", 22.5), ("NVIDIA A100-SXM4-80GB", 80))] == ["t4", "l4", "a100"]
    # credentials cell: env-var and file fallbacks both feed ~/.kaggle
    cred_src = code[3]
    assert "KAGGLE_CREDENTIALS" in cred_src and 'WORK / "credentials.json"' in cred_src
    with tempfile.TemporaryDirectory() as tmp:
        ns_c = {"os": os, "json": __import__("json"), "pathlib": __import__("pathlib"), "stat": __import__("stat"),
                "WORK": pathlib.Path(tmp), "OWNER": "me", "SystemExit": SystemExit}
        (pathlib.Path(tmp) / "credentials.json").write_text(json.dumps({"username": "me", "key": "k"}))
        home = pathlib.Path(tmp) / "h"
        home.mkdir()
        with mock.patch("pathlib.Path.home", return_value=home):
            exec(compile(cred_src, "creds", "exec"), ns_c)
        assert json.loads((home / ".kaggle" / "kaggle.json").read_text())["key"] == "k"
        # files cell restores every embedded file byte for byte under WORK
        ns2 = {"pathlib": __import__("pathlib"), "print": lambda *a, **k: None,
               "WORK": pathlib.Path(tmp) / "ws"}
        exec(code[4], ns2)
        REPO = os.path.dirname(os.path.dirname(HERE))
        for rel in EMBED:
            assert open(os.path.join(tmp, "ws", "repo", rel), "rb").read() == open(
                os.path.join(REPO, rel), "rb").read(), rel
    # same embedded payload as the Colab notebook, bit for bit
    import make_colab_sft_nb as colab
    assert embed() == colab.embed()
    # the platform-dependent cells differ, the shared ones are byte-identical to Colab's
    cc = ["".join(c["source"]) for c in colab.build()["cells"] if c["cell_type"] == "code"]
    for idx in (1, 2, 5, 7, 8):                      # gpu / deps / base / train / gate
        assert code[idx] == cc[idx], idx
    for idx in (0, 3, 4, 6, 9, 10):                  # config / creds / files / data / upload / verify
        assert code[idx] != cc[idx], idx
    size = len(json.dumps(nb))
    assert size < 2_000_000
    print("selftest ok: 11 code cells compile; Lightning secrets/file credentials; WORK-persistent paths; "
          "shared cells byte-identical to the Colab notebook; files cell restores all %d embedded files; "
          "notebook size %.2f MB" % (len(EMBED), size / 1e6))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--selftest"]:
        selftest()
        sys.exit(0)
    opt = {k: a[a.index(k) + 1] for k in ("--out", "--owner", "--slug") if k in a}
    out = opt.get("--out", os.path.join(os.path.dirname(HERE), "lightning_sft_ag2.ipynb"))
    nb = build(opt.get("--owner", "takumuhata"), opt.get("--slug", "qwen3-4b-grids15-sft-a"))
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print("wrote %s (%.2f MB, %d cells). Embedded:" % (out, os.path.getsize(out) / 1e6, len(nb["cells"])))
    for rel, (sha, _) in embed().items():
        print("  %s  %s" % (sha[:16], rel))
