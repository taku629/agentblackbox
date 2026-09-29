# ARC-AGI-3 STaR LoRA eval — Colab setup script
# Ported from bundle setup_commands.json: same vLLM server flags, but vLLM is
# installed from PyPI (cell 1) instead of the Kaggle wheelhouse dataset, and
# dataset paths resolve from TAAF_KAGGLE_INPUT_PATHS into /content.
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL_OWNER = 'rahim3'
MODEL_SLUG = 'qwen3-8-27b-bf16'
SERVED_MODEL_NAME = 'duck-27b-lora'
VLLM_HOST = '127.0.0.1'
VLLM_PORT = 1234
VLLM_BASE_URL = f'http://{VLLM_HOST}:{VLLM_PORT}/v1'
VLLM_MAX_MODEL_LEN = 65536
ANALYZER_CONTEXT_WINDOW = 32768
VLLM_TENSOR_PARALLEL_SIZE = 1
WORKING_DIR = Path(os.environ['TAAF_KAGGLE_WORKING_DIR'])
SITE_PACKAGES = WORKING_DIR / 'vllm-site-packages'
VLLM_SERVER_LOG = WORKING_DIR / 'vllm-openai-server.log'
VLLM_SERVER_PID = WORKING_DIR / 'vllm-openai-server.pid'


def taaf_kaggle_input_paths() -> dict:
    raw = os.getenv('TAAF_KAGGLE_INPUT_PATHS', '').strip()
    if not raw:
        return {}
    data = json.loads(raw)
    return {str(ref): Path(str(path)) for ref, path in data.items()}


def resolve_kaggle_dataset_path(owner: str, slug: str) -> Path:
    mapped = taaf_kaggle_input_paths().get(f'{owner}/{slug}')
    if mapped is not None:
        return mapped
    return Path('/content') / slug


MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)
LORA_PATH = resolve_kaggle_dataset_path('takumuhata', 'taaf-duck-lora-v1')


def vllm_env() -> dict:
    env = os.environ.copy()
    env['VLLM_NO_USAGE_STATS'] = '1'
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    env['PYTHONPATH'] = str(SITE_PACKAGES) + os.pathsep + env.get('PYTHONPATH', '')
    return env


def request_json(url: str, *, payload=None, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def verify_vllm() -> None:
    result = subprocess.run(
        [sys.executable, '-c', 'import vllm; print(vllm.__version__)'],
        capture_output=True, text=True,
    )
    print('vLLM check:', (result.stdout or result.stderr).strip(), flush=True)
    if result.returncode != 0:
        raise RuntimeError('vLLM import failed; run the pip install cell first.')


def wait_for_vllm_server() -> None:
    deadline = time.monotonic() + 900.0
    last_error = ''
    while time.monotonic() < deadline:
        try:
            payload = request_json(f'{VLLM_BASE_URL}/models', timeout=10)
            names = [entry.get('id') for entry in payload.get('data', [])]
            if SERVED_MODEL_NAME in names:
                print(f'vLLM server ready: {names}', flush=True)
                return
            last_error = f'models endpoint returned {names}'
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(10)
    tail = ''
    if VLLM_SERVER_LOG.exists():
        tail = VLLM_SERVER_LOG.read_text(encoding='utf-8', errors='replace')[-4000:]
    raise RuntimeError(f'vLLM server did not become ready: {last_error}\n{tail}')


def start_vllm_server() -> None:
    verify_vllm()
    VLLM_SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    VLLM_SERVER_PID.unlink(missing_ok=True)
    log_handle = VLLM_SERVER_LOG.open('w', encoding='utf-8')
    cmd = [
        sys.executable,
        '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(MODEL_PATH),
        '--served-model-name', 'Qwen/Qwen3.8-27B-bf16',
        '--enable-lora',
        '--max-lora-rank', '16',
        '--lora-modules', SERVED_MODEL_NAME + '=' + str(LORA_PATH),
        '--host', VLLM_HOST,
        '--port', str(VLLM_PORT),
        '--tensor-parallel-size', str(VLLM_TENSOR_PARALLEL_SIZE),
        '--enable-auto-tool-choice',
        '--tool-call-parser', 'qwen3_coder',
        '--generation-config', 'vllm',
        '--enable-prefix-caching',
        '--default-chat-template-kwargs', '{"preserve_thinking": true}',
        '--reasoning-parser', 'qwen3',
        '--max-model-len', str(VLLM_MAX_MODEL_LEN),
    ]
    extra = os.getenv('VLLM_EXTRA_ARGS', '').strip()
    if extra:
        cmd.extend(extra.split())
    print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)
    process = subprocess.Popen(cmd, env=vllm_env(), stdout=log_handle,
                               stderr=subprocess.STDOUT, text=True)
    VLLM_SERVER_PID.write_text(str(process.pid), encoding='utf-8')
    wait_for_vllm_server()


def run_vllm_api_smoke_test() -> None:
    payload = {
        'model': SERVED_MODEL_NAME,
        'messages': [{'role': 'user', 'content': 'Answer in one short sentence: what is 2 + 2?'}],
        'temperature': 0.0,
        'max_tokens': 96,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    response = request_json(f'{VLLM_BASE_URL}/chat/completions', payload=payload, timeout=120)
    generated = response['choices'][0]['message'].get('content', '').strip()
    print('\n' + '=' * 88, flush=True)
    print('VLLM OPENAI SERVER QWEN SMOKE TEST REAL MODEL OUTPUT', flush=True)
    print('Generated:', generated, flush=True)
    print('=' * 88 + '\n', flush=True)


print(f'Qwen model path: {MODEL_PATH}', flush=True)
print(f'LoRA path: {LORA_PATH}', flush=True)
assert shutil.which('nvidia-smi'), 'CUDA GPU check failed: nvidia-smi is not available.'
missing = [str(path) for path in (MODEL_PATH, LORA_PATH) if not path.exists()]
if missing:
    raise FileNotFoundError('Missing attached dataset path(s): ' + ', '.join(missing))
start_vllm_server()
run_vllm_api_smoke_test()
setup_env = {
    'USE_TF': '0',
    'TRANSFORMERS_NO_TF': '1',
    'TRANSFORMERS_NO_TORCHVISION': '1',
    'VLLM_NO_USAGE_STATS': '1',
    'PYTHONPATH': str(SITE_PACKAGES) + os.pathsep + os.environ.get('PYTHONPATH', ''),
    'LOCAL_ANALYZER_BASE_URL': VLLM_BASE_URL,
    'OPENAI_BASE_URL': VLLM_BASE_URL,
    'LOCAL_ANALYZER_PROVIDER': 'vllm',
    'OPENAI_PROVIDER': 'vllm',
    'LOCAL_ANALYZER_MODEL_ID': SERVED_MODEL_NAME,
    'INFERENCE_ANALYZER_MODEL': SERVED_MODEL_NAME,
    'LOCAL_ANALYZER_APP_NAME': 'ARC3 Agent Harness',
    'LOCAL_ANALYZER_CONTEXT_WINDOW': str(ANALYZER_CONTEXT_WINDOW),
    'LOCAL_ANALYZER_MAX_OUTPUT': '6144',
    'LOCAL_ANALYZER_TOOL_STEPS': '0',
    'LOCAL_ANALYZER_TOOL_TIMEOUT': '30',
    'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': '1024',
    'LOCAL_ANALYZER_YIELD_SECONDS': '60',
    'LOCAL_ANALYZER_TEMPERATURE': '0.6',
    'LOCAL_ANALYZER_TOP_P': '0.95',
    'LOCAL_ANALYZER_TOP_K': '20',
    'LOCAL_ANALYZER_ENABLE_THINKING': 'true',
    'MULTIMODAL_CONTEXT': 'current_grid',
    'MULTIMODAL_UPSCALE': '4',
}
setup_env_path = Path(os.environ['TAAF_KAGGLE_SETUP_ENV'])
existing_setup_env = {}
if setup_env_path.exists():
    existing_setup_env = json.loads(setup_env_path.read_text(encoding='utf-8'))
    if not isinstance(existing_setup_env, dict):
        raise RuntimeError('TAAF_KAGGLE_SETUP_ENV must contain a JSON object.')
existing_setup_env.update(setup_env)
setup_env_path.write_text(json.dumps(existing_setup_env, indent=2), encoding='utf-8')
print('setup env written:', setup_env_path, flush=True)
