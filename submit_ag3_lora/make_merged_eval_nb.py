"""Build colab_merged_eval.ipynb — merged-model eval of the QLoRA adapter.

Variant of colab_lora_eval.ipynb: instead of serving bf16 base + --enable-lora
(the mismatched path for an NF4-trained adapter), it first runs merge_setup.py
(peft merge_and_unload on the 4-bit base -> bf16 merged model = training-time
effective weights) and serves /content/merged_27b under the duck-27b-lora name.

Subset of 7 signal games by default (MERGED_EVAL_FULL=1 for all 25).
Verdict rule: merged-subset-mean >> 1.669-equivalent -> adapter fine (mismatch
was the cause); still ~1.7 -> adapter bad, retrain in bf16.
"""
import json
from pathlib import Path

SETUP_SRC = Path(__file__).with_name("colab_eval_setup.py").read_text()
MERGE_SRC = Path(__file__).with_name("merge_setup.py").read_text()
assert "'''" not in SETUP_SRC and "'''" not in MERGE_SRC

# Reuse the exact cells from the lora-eval builder for identical behavior.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "meb", str(Path(__file__).with_name("make_eval_nb.py")))
meb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meb)  # side effect: rewrites colab_lora_eval.ipynb (idempotent)

CELLS = []

CELLS.append(("md", """# ARC-AGI-3 — QLoRA adapter merged-model eval (diagnostic)

The QLoRA adapter was trained on the 4-bit NF4 base but the first eval served
**bf16 base + adapter** — the mismatched path (adapter absorbed NF4 quant
corrections). That eval scored **mean 1.669** vs un-adapted 27B **4.97** —
the adapter actively hurt the model.

This notebook tests the *proper* QLoRA path: **dequantize NF4 base + merge
deltas** -> merged bf16 model -> serve directly.

- merged subset mean ≈ base (≈5) or better -> adapter is fine; production =
  merged model (Kaggle, no --enable-lora needed)
- still ≈1.7 -> adapter itself is bad -> retrain in bf16

**Prereqs**: A100 80GB runtime, Colab Secret `KAGGLE_CREDENTIALS`.
Download ~50min + merge ~20min + 7-game eval ~1h. Set MERGED_EVAL_FULL=1 in the
games cell for all 25.
"""))

# cells 1-4 identical to the lora eval: pip/venv, creds, downloads, unpack deps
CELLS += meb.CELLS[1:5]

CELLS.append(("code", """# Merge: dequantize NF4 base + add LoRA deltas -> /content/merged_27b (bf16).
MERGE_SRC = r'''
""" + MERGE_SRC + """
'''
pathlib.Path('/content/merge_setup.py').write_text(MERGE_SRC)
r = subprocess.run([sys.executable, '/content/merge_setup.py'], cwd='/content')
assert r.returncode == 0, 'merge failed'
assert pathlib.Path('/content/merged_27b/model.safetensors.index.json').exists()
print('merged model ready')
"""))

CELLS.append(("code", """# TAAF env wiring — identical to the lora eval, plus MERGED_MODEL_PATH.
import json, os, pathlib, sys

BUNDLE_DIR = pathlib.Path('/content/bundle')
WORKING_DIR = pathlib.Path('/content/working'); WORKING_DIR.mkdir(exist_ok=True)
SETUP_ENV_PATH = WORKING_DIR / 'taaf_setup_env.json'

kaggle_input_paths = {
    'takumuhata/taaf-anim-27b-lora': str(BUNDLE_DIR),
    'driessmit1/arc3-vllm-h100-wheelhouse-v3': str(BUNDLE_DIR),
    'rahim3/qwen3-8-27b-bf16': '/content/base_27b',
    'takumuhata/taaf-duck-lora-v1': '/content/lora_adapter',
}
setup_env = {
    'TAAF_KAGGLE_INPUT_PATHS': json.dumps(kaggle_input_paths, sort_keys=True),
    'TAAF_KAGGLE_BUNDLE_DIR': str(BUNDLE_DIR),
    'TAAF_KAGGLE_WORKING_DIR': str(WORKING_DIR),
    'TAAF_KAGGLE_SETUP_ENV': str(SETUP_ENV_PATH),
    'TAAF_QWEN_MODEL_REF': 'merged-27b-qlora',
    'TAAF_QWEN_MODEL_PATH': '/content/merged_27b',
    'TAAF_QWEN_SERVED_MODEL_NAME': 'duck-27b-lora',
    'MERGED_MODEL_PATH': '/content/merged_27b',
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'TAAF_RUN_AS_SUBMISSION': '0',
    'TAAF_MINIMAL_DIAGNOSTICS': '0',
    'ONLY_RESET_LEVELS': 'true',
    'MPLBACKEND': 'Agg',
}
os.environ.update(setup_env)
SETUP_ENV_PATH.write_text(json.dumps(setup_env, indent=2))

for repo in sorted((BUNDLE_DIR / 'src').iterdir(), reverse=True):
    if repo.is_dir():
        for cand in (repo / 'src', repo):
            if cand.is_dir() and str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
if ARC_PKGS not in sys.path:
    sys.path.insert(0, ARC_PKGS)
import taaf  # noqa: F401
print('taaf OK; serving merged model at', os.environ['MERGED_MODEL_PATH'])
"""))

CELLS.append(("code", "SETUP_SRC = r'''\n" + SETUP_SRC + "\n'''\n"
"""pathlib.Path('/content/colab_eval_setup.py').write_text(SETUP_SRC)
env = dict(os.environ); env['PYTHON'] = sys.executable
r = subprocess.run([sys.executable, '/content/colab_eval_setup.py'],
                   env=env, cwd=str(WORKING_DIR))
assert r.returncode == 0, 'vllm setup failed'
os.environ.update(json.loads(SETUP_ENV_PATH.read_text()))
"""))

CELLS.append(("code", """# Benchmark override — 7 signal games by default; MERGED_EVAL_FULL=1 -> 25.
import pickle
with open(BUNDLE_DIR / 'deploy_target.pkl', 'rb') as f:
    target = pickle.load(f)
target.actual_run_as_submission = False
target.is_competition_rerun = False
with open(BUNDLE_DIR / 'benchmark_initial.pkl', 'rb') as f:
    bm = pickle.load(f)
bm.job_dir = WORKING_DIR

# 7 games with signal across both prior evals (k27v2 base=4.97 mean,
# lora-eval=1.669): lora-eval scored on all of these except ar25/dc22.
SUBSET = [
    'ft09-0d8bbf25',  # base 47.6, lora 14.3 — biggest regression
    'lp85-305b61c3',  # base 16.7, lora 2.8
    'ar25-0c556536',  # base 8.3,  lora 0.0
    'sc25-635fd71a',  # base 6.1,  lora 0.0
    're86-8af5384d',  # base 6.4,  lora 2.1
    'vc33-5430563c',  # base 9.7,  lora 10.7 (lora WON here)
    'cn04-2fe56bfb',  # base 2.3,  lora 4.8  (lora won)
]
FULL = [
    'ar25-0c556536', 'bp35-0a0ad940', 'cd82-fb555c5d', 'cn04-2fe56bfb',
    'dc22-fdcac232', 'ft09-0d8bbf25', 'g50t-5849a774', 'ka59-38d34dbb',
    'lf52-271a04aa', 'lp85-305b61c3', 'ls20-9607627b', 'm0r0-492f87ba',
    'r11l-495a7899', 're86-8af5384d', 's5i5-18d95033', 'sb26-7fbdac44',
    'sc25-635fd71a', 'sk48-d8078629', 'sp80-589a99af', 'su15-1944f8ab',
    'tn36-ef4dde99', 'tr87-cd924810', 'tu93-0768757b', 'vc33-5430563c',
    'wa30-ee6fef47',
]
import os as _os
GAMES = FULL if _os.environ.get('MERGED_EVAL_FULL') == '1' else SUBSET

import taaf.game_api
spec = bm.games[0].arcade_spec
object.__setattr__(spec, 'environments_dir', pathlib.Path(ENV_DIR))
bm.games = [taaf.game_api.GameAPI(env_name=g, arcade_spec=spec)
            for g in GAMES]
bm.n_passes = 1
bm.game_weights = None
bm.label = f'{bm.label}-merged-{len(GAMES)}g-p1-colab'
print(f'{len(bm.games)} games x 1 pass; env_dir={ENV_DIR}')
"""))

CELLS.append(("code", """# Run eval + teardown (same as lora eval).
import contextlib, sys
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
    def flush(self):
        for st in self.streams: st.flush()
_log = open(WORKING_DIR / 'stdout.log', 'w', encoding='utf-8')
_orig = sys.stdout; sys.stdout = _Tee(sys.stdout, _log)
try:
    await bm.run(soft_end_time=None, runtime_environment=target,
                 minimal_diagnostics=False)
finally:
    sys.stdout = _orig; _log.close()
    pid_path = WORKING_DIR / 'vllm-openai-server.pid'
    if pid_path.exists():
        import signal, time
        pid = int(pid_path.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM); time.sleep(10)
        except Exception as e:
            print('teardown:', repr(e))
print('eval done')
"""))

CELLS.append(("code", """# Results + verdict vs both prior evals.
import json, pathlib
d = json.loads((WORKING_DIR / 'benchmark.json').read_text())
runs = d['game_runs']
BASE = {'ft09-0d8bbf25': 47.62, 'lp85-305b61c3': 16.67, 'ar25-0c556536': 8.33,
        'sc25-635fd71a': 6.12, 're86-8af5384d': 6.43, 'vc33-5430563c': 9.66,
        'cn04-2fe56bfb': 2.27}
LORA = {'ft09-0d8bbf25': 14.29, 'lp85-305b61c3': 2.78, 'ar25-0c556536': 0.0,
        'sc25-635fd71a': 0.0, 're86-8af5384d': 2.09, 'vc33-5430563c': 10.71,
        'cn04-2fe56bfb': 4.76}
print(f"{'game':18} {'base':>8} {'lora-eval':>10} {'merged':>8}")
tot_b = tot_l = tot_m = 0.0
for r in runs:
    g = r['game_id']; m = r.get('final_score') or 0.0
    b = BASE.get(g); l = LORA.get(g)
    tot_b += b or 0; tot_l += l or 0; tot_m += m
    print(f"{g:18} {b if b is not None else '-':>8} "
          f"{l if l is not None else '-':>10} {m:>8.2f}")
n = len(runs)
print(f"{'means':18} {tot_b/n:>8.2f} {tot_l/n:>10.2f} {tot_m/n:>8.2f}")
out = {'mean': tot_m / n, 'n': n, 'base_subset_mean': tot_b / n,
       'lora_subset_mean': tot_l / n,
       'verdict': ('adapter fine — mismatch was the cause'
                   if tot_m / n > tot_l / n * 1.5 else 'adapter bad — retrain bf16'),
       'runs': [(r['game_id'], r.get('final_score'), r.get('state'))
                for r in runs]}
pathlib.Path('/content/merged_eval_result.json').write_text(json.dumps(out, indent=2))
print('VERDICT:', out['verdict'])
"""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "name": "colab_merged_eval.ipynb"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "accelerator": "GPU", "gpuClass": "a100",
    },
    "cells": [],
}
for kind, src in CELLS:
    cell = {"metadata": {}, "source": src.splitlines(keepends=True),
            "cell_type": "markdown" if kind == "md" else "code"}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    nb["cells"].append(cell)

out = Path(__file__).with_name("colab_merged_eval.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, len(nb["cells"]), "cells")
