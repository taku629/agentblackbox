"""Build colab_lora_eval.ipynb — Colab port of arc3-duck-anim-27b-lora eval."""
import json
from pathlib import Path

SETUP_SRC = Path(__file__).with_name("colab_eval_setup.py").read_text()
assert "'''" not in SETUP_SRC, "setup script must not contain triple quotes"

CELLS = []

CELLS.append(("md", """# ARC-AGI-3 — STaR LoRA eval on Colab (port of arc3-duck-anim-27b-lora)

Runs the 25 public games × 1 pass offline eval: vLLM bf16 `qwen3-8-27b-bf16` +
LoRA adapter `takumuhata/taaf-duck-lora-v1` served as `duck-27b-lora`.

**Prereqs**
- Runtime: **A100 80GB** (bf16 27B ≈ 54 GB weights). Runtime → Change runtime type → A100.
- Colab Secret `KAGGLE_CREDENTIALS` = contents of `~/.kaggle/credentials.json` for
  **takumuhata** (needed for the private bundle/adapter/deps datasets).

Run all cells. Download phase ≈ 40–60 min (44 GB base model), eval ≈ 6–8 h.
"""))

CELLS.append(("code", """import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',
                       'vllm==0.19.0', 'kaggle==1.7.4.5',
                       'python-dotenv', 'imageio', 'scipy', 'matplotlib',
                       'pillow', 'pandas', 'pyarrow'])
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',
                           'flashinfer-python==0.6.6'])
except Exception as exc:
    print('flashinfer install failed (vllm falls back):', repr(exc)[:200])
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
assert torch.cuda.is_available(), 'pick a GPU runtime (A100 80GB required)'
gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print('GPU VRAM GB:', gb)
assert gb > 60, 'bf16 27B needs ~54GB weights — use A100 80GB'
import vllm
print('vllm', vllm.__version__)
assert vllm.__version__.startswith('0.19'), 'expected vllm 0.19.x'
"""))

CELLS.append(("code", """import os, pathlib, shutil, subprocess
kd = pathlib.Path.home() / '.kaggle'; kd.mkdir(exist_ok=True)
cred_path = kd / 'credentials.json'
# Preferred: Colab secret. Fallback: credentials.json uploaded to /content.
try:
    from google.colab import userdata
    cred = userdata.get('KAGGLE_CREDENTIALS')
except Exception:
    cred = None
if cred and 'takumuhata' in cred:
    cred_path.write_text(cred)
elif pathlib.Path('/content/kaggle_credentials.json').exists():
    shutil.copy('/content/kaggle_credentials.json', cred_path)
else:
    raise AssertionError(
        'set Colab secret KAGGLE_CREDENTIALS, or upload credentials.json '
        'to /content/kaggle_credentials.json via the Files panel')
os.chmod(cred_path, 0o600)
assert 'takumuhata' in cred_path.read_text(), 'credentials are not takumuhata'
import json as _json, requests as _rq
_cred = _json.loads(cred_path.read_text())
_r = _rq.post('https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset',
              json={'ownerSlug': 'takumuhata', 'datasetSlug': 'taaf-colab-deps-v1'},
              headers={'Authorization': 'Bearer ' + _cred['access_token']},
              stream=True, allow_redirects=False)
print('auth check status:', _r.status_code)
_r.close()
assert _r.status_code in (200, 301, 302), 'kaggle auth failed — check credentials'
"""))

CELLS.append(("code", """# Stream-download Kaggle datasets via the real API flow:
# POST api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset with a
# Bearer access_token -> 302 -> signed GCS URL. The stored access_token can be
# expired; refresh it via /api/v1/access-tokens/generate (refresh_token).
# Legacy GET www.kaggle.com/api/v1/datasets/download/<slug> works for PUBLIC
# datasets only and is kept as fallback. CLI is broken on this runtime.
import json, pathlib, requests, zipfile, sys, time
_KJ = pathlib.Path.home() / '.kaggle' / 'credentials.json'
_CRED = json.loads(_KJ.read_text())
_TOK = _CRED['access_token']
_API = 'https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset'
_LEGACY = 'https://www.kaggle.com/api/v1/datasets/download/'

def _refresh():
    global _TOK
    r = requests.post('https://www.kaggle.com/api/v1/access-tokens/generate',
        json={'refreshToken': _CRED['refresh_token'], 'apiVersion': 'API_VERSION_V1'},
        headers={'Authorization': 'Bearer ' + _TOK, 'Content-Type': 'application/json'},
        timeout=60)
    if r.status_code == 200:
        _TOK = (r.json().get('token') or r.json().get('accessToken')) or _TOK
        _CRED['access_token'] = _TOK
        _KJ.write_text(json.dumps(_CRED))
        print('token refreshed', flush=True)
    else:
        print('token refresh ->', r.status_code, flush=True)

def dl(slug, dest, marker):
    dest = pathlib.Path(dest); dest.mkdir(exist_ok=True)
    if (dest / marker).exists():
        print(slug, 'already present'); return
    owner, name = slug.split('/')
    zp = dest / (name + '.zip')
    print('downloading', slug, flush=True)
    got = False
    for attempt in range(3):
        try:
            r = requests.post(_API, json={'ownerSlug': owner, 'datasetSlug': name},
                headers={'Authorization': 'Bearer ' + _TOK, 'Content-Type': 'application/json'},
                stream=True, timeout=120)
            if r.status_code in (401, 403) and attempt == 0:
                r.close(); _refresh(); continue
            if r.status_code in (401, 403):
                r.close()
                r = requests.get(_LEGACY + slug,
                    headers={'Authorization': 'Bearer ' + _TOK}, stream=True, timeout=120)
            r.raise_for_status()
            with open(zp, 'wb') as f:
                n = 0
                for chunk in r.iter_content(1 << 22):
                    if chunk:
                        f.write(chunk); n += len(chunk)
                        if n % (1 << 30) < (1 << 22):
                            print('  %.1f GB' % (n / 1e9), flush=True)
            print('downloaded %.2f GB' % (n / 1e9), flush=True)
            got = True
            break
        except Exception as e:
            print('attempt', attempt, 'failed:', e, flush=True)
            time.sleep(20); _refresh()
    if not got:
        sys.exit('FAILED ' + slug)
    if zipfile.is_zipfile(zp):
        print('unzipping...', flush=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(dest)
        zp.unlink()
    else:
        print('not a zip, leaving raw file at', zp)

dl('takumuhata/taaf-colab-deps-v1', '/content/deps', 'arc_pkgs.zip')
dl('takumuhata/taaf-anim-27b-lora', '/content/bundle', 'taaf-kaggle-bundle.json')
dl('takumuhata/taaf-duck-lora-v1', '/content/lora_adapter', 'adapter_model.safetensors')
dl('rahim3/qwen3-8-27b-bf16', '/content/base_27b', 'config.json')
print('base shards:', len(list(pathlib.Path('/content/base_27b').glob('*.safetensors'))))
"""))

CELLS.append(("code", """# Unpack nested zips in the deps dataset; locate arc pkgs + env files.
# NOTE: the deps dataset may hold plain dirs (arc_pkgs/, environment_files/)
# rather than nested zips — search both roots.
import glob, pathlib, sys, zipfile
for z in sorted(glob.glob('/content/deps/*.zip')):
    with zipfile.ZipFile(z) as zf:
        zf.extractall('/content/deps_unz')

def _find(name):
    for root in ('/content/deps_unz', '/content/deps'):
        hits = glob.glob(root + '/**/' + name, recursive=True)
        if hits:
            return pathlib.Path(hits[0]).parent
    raise FileNotFoundError(name)

ARC_PKGS = str(_find('arcengine'))
ENV_DIR = str(_find('ar25'))
print('ARC_PKGS:', ARC_PKGS)
print('ENV_DIR:', ENV_DIR, '→', len(list(pathlib.Path(ENV_DIR).iterdir())), 'dirs')
if ARC_PKGS not in sys.path:
    sys.path.insert(0, ARC_PKGS)
import arcengine, arc_agi
print('arcengine OK:', getattr(arcengine, '__version__', '?'))
"""))

CELLS.append(("code", """# TAAF env wiring — mirrors kernel cell 3/5, /kaggle/* → /content/*.
import json, os, pathlib, sys

BUNDLE_DIR = pathlib.Path('/content/bundle')
WORKING_DIR = pathlib.Path('/content/working'); WORKING_DIR.mkdir(exist_ok=True)
SETUP_ENV_PATH = WORKING_DIR / 'taaf_setup_env.json'

QWEN_MODEL_PATH = pathlib.Path('/content/base_27b')
for name in ('config.json', 'model.safetensors.index.json',
             'tokenizer.json', 'tokenizer_config.json'):
    assert (QWEN_MODEL_PATH / name).is_file(), f'missing {name}'
assert len(list(QWEN_MODEL_PATH.glob('model-*.safetensors'))) == 18

kaggle_input_paths = {
    'takumuhata/taaf-anim-27b-lora': str(BUNDLE_DIR),
    'driessmit1/arc3-vllm-h100-wheelhouse-v3': str(BUNDLE_DIR),  # unused on Colab
    'rahim3/qwen3-8-27b-bf16': str(QWEN_MODEL_PATH),
    'takumuhata/taaf-duck-lora-v1': '/content/lora_adapter',
}
setup_env = {
    'TAAF_KAGGLE_INPUT_PATHS': json.dumps(kaggle_input_paths, sort_keys=True),
    'TAAF_KAGGLE_BUNDLE_DIR': str(BUNDLE_DIR),
    'TAAF_KAGGLE_WORKING_DIR': str(WORKING_DIR),
    'TAAF_KAGGLE_SETUP_ENV': str(SETUP_ENV_PATH),
    'TAAF_QWEN_MODEL_REF': 'rahim3/qwen3-8-27b-bf16',
    'TAAF_QWEN_MODEL_PATH': str(QWEN_MODEL_PATH),
    'TAAF_QWEN_SERVED_MODEL_NAME': 'duck-27b-lora',
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'TAAF_RUN_AS_SUBMISSION': '0',
    'TAAF_MINIMAL_DIAGNOSTICS': '0',
    'ONLY_RESET_LEVELS': 'true',
    'MPLBACKEND': 'Agg',
}
os.environ.update(setup_env)
SETUP_ENV_PATH.write_text(json.dumps(setup_env, indent=2))

# Bundle repos + arcengine/arc_agi importable.
for repo in sorted((BUNDLE_DIR / 'src').iterdir(), reverse=True):
    if repo.is_dir():
        for cand in (repo / 'src', repo):
            if cand.is_dir() and str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
if ARC_PKGS not in sys.path:
    sys.path.insert(0, ARC_PKGS)
import taaf  # noqa: F401
print('taaf OK; PYTHONPATH entries:', len(sys.path))
"""))

CELLS.append(("code", "SETUP_SRC = r'''\n" + SETUP_SRC + "\n'''\n"
"""pathlib.Path('/content/colab_eval_setup.py').write_text(SETUP_SRC)
env = dict(os.environ); env['PYTHON'] = sys.executable
r = subprocess.run([sys.executable, '/content/colab_eval_setup.py'],
                   env=env, cwd=str(WORKING_DIR))
assert r.returncode == 0, 'vllm setup failed'
# Apply the setup env written by the server bootstrap.
os.environ.update(json.loads(SETUP_ENV_PATH.read_text()))
"""))

CELLS.append(("code", """# Load benchmark + deploy pickles, override to 25 public games x 1 pass.
import pickle
with open(BUNDLE_DIR / 'deploy_target.pkl', 'rb') as f:
    target = pickle.load(f)
target.actual_run_as_submission = False
target.is_competition_rerun = False
with open(BUNDLE_DIR / 'benchmark_initial.pkl', 'rb') as f:
    bm = pickle.load(f)
bm.job_dir = WORKING_DIR

Q38_P1_PUBLIC_GAME_IDS = [
    'ar25-0c556536', 'bp35-0a0ad940', 'cd82-fb555c5d', 'cn04-2fe56bfb',
    'dc22-fdcac232', 'ft09-0d8bbf25', 'g50t-5849a774', 'ka59-38d34dbb',
    'lf52-271a04aa', 'lp85-305b61c3', 'ls20-9607627b', 'm0r0-492f87ba',
    'r11l-495a7899', 're86-8af5384d', 's5i5-18d95033', 'sb26-7fbdac44',
    'sc25-635fd71a', 'sk48-d8078629', 'sp80-589a99af', 'su15-1944f8ab',
    'tn36-ef4dde99', 'tr87-cd924810', 'tu93-0768757b', 'vc33-5430563c',
    'wa30-ee6fef47',
]
import taaf.game_api
spec = bm.games[0].arcade_spec
# spec is a frozen attrs instance — setattr guard, assign via object.__setattr__
object.__setattr__(spec, 'environments_dir', pathlib.Path(ENV_DIR))  # Colab env files, not /kaggle/input
bm.games = [taaf.game_api.GameAPI(env_name=g, arcade_spec=spec)
            for g in Q38_P1_PUBLIC_GAME_IDS]
bm.n_passes = 1
bm.game_weights = None
bm.label = f'{bm.label}-25g-p1-colab'
print(f'{len(bm.games)} games x {bm.n_passes} pass; env_dir={ENV_DIR}')
print('concurrency:', getattr(bm.solver, 'concurrency', None),
      'per-game cap s:', getattr(bm.solver, 'max_runtime_s_per_game', None))
"""))

CELLS.append(("code", """# Run the eval (25 games x 1 pass, ~6-8h on A100). Stdout tees to stdout.log.
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
    # teardown: stop vllm server
    pid_path = WORKING_DIR / 'vllm-openai-server.pid'
    if pid_path.exists():
        import signal, time
        pid = int(pid_path.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM); time.sleep(10)
        except Exception as e:
            print('teardown:', repr(e))
print('eval done — see next cell for scores')
"""))

CELLS.append(("code", """# Results: mean final_score across the 25 public games.
import json, pathlib
bj = WORKING_DIR / 'benchmark.json'
d = json.loads(bj.read_text())
runs = d['game_runs']
scores = [(g['game_id'], g.get('final_score'), g.get('state'),
           g.get('levels_completed'), g.get('number_of_levels')) for g in runs]
vals = [s for _, s, *_ in scores if s is not None]
print(f'mean final_score over {len(vals)} games: {sum(vals)/len(vals):.3f}')
for g, s, st, lc, nl in sorted(scores, key=lambda t: -(t[1] or 0)):
    print(f'  {g}: {s if s is not None else \"?\":>7} ({st}, lv {lc}/{nl})')
out = {'mean': sum(vals)/len(vals), 'n': len(vals), 'runs': scores}
pathlib.Path('/content/lora_eval_result.json').write_text(json.dumps(out, indent=2))
print('saved /content/lora_eval_result.json')
"""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "name": "colab_lora_eval.ipynb"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "accelerator": "GPU",
        "gpuClass": "a100",
    },
    "cells": [],
}
for kind, src in CELLS:
    cell = {"metadata": {}, "source": src.splitlines(keepends=True),
            "cell_type": "markdown" if kind == "md" else "code"}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    nb["cells"].append(cell)

out = Path(__file__).with_name("colab_lora_eval.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, len(nb["cells"]), "cells")
