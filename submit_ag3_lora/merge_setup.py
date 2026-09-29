# Proper QLoRA merge: dequantize the NF4 base (same quant config the adapter
# was trained against) and add the LoRA deltas, yielding a bf16 merged model.
# Serving bf16-base + adapter directly is the MISMATCHED path (adapter absorbed
# NF4 corrections); merged = W_nf4 + Delta reproduces training-time weights.
import os, pathlib, sys, torch

BASE = pathlib.Path('/content/base_27b')
ADAPT = pathlib.Path('/content/lora_adapter')
MERGED = pathlib.Path('/content/merged_27b')

assert (BASE / 'model.safetensors.index.json').exists(), 'base model missing'
assert (ADAPT / 'adapter_model.safetensors').exists(), 'adapter missing'
assert torch.cuda.is_available(), 'bitsandbytes dequantize needs CUDA'

if (MERGED / 'model.safetensors.index.json').exists():
    print('merged already present:', MERGED); sys.exit(0)

from transformers import AutoModelForImageTextToText, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
print('loading base in 4-bit NF4 (training-time config)...', flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    str(BASE), quantization_config=bnb, device_map='auto')
print('attaching adapter...', flush=True)
model = PeftModel.from_pretrained(model, str(ADAPT))
print('merge_and_unload (dequantize + add deltas)...', flush=True)
model = model.merge_and_unload()
MERGED.mkdir(exist_ok=True)
model.save_pretrained(str(MERGED), safe_serialization=True)
tok = AutoTokenizer.from_pretrained(str(BASE))
tok.save_pretrained(str(MERGED))
# adapter ships a chat_template.jinja too — keep the base tokenizer's;
# eval harness controls templating anyway. Copy aux configs if absent.
for name in ('generation_config.json',):
    src = BASE / name
    if src.exists() and not (MERGED / name).exists():
        (MERGED / name).write_text(src.read_text())
print('merged saved ->', MERGED,
      'shards:', len(list(MERGED.glob('*.safetensors'))), flush=True)
