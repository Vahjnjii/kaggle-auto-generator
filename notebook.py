import subprocess
import sys

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", 
                      "diffusers", "transformers", "accelerate", "peft", "-q"])

import torch
import random
import gc
import os
import zipfile
from datetime import datetime
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

print("=" * 70)
print("GITHUB ACTIONS AUTO-GENERATOR")
print("=" * 70)

PROMPTS_TEXT = """PROMPTS_PLACEHOLDER"""

SUB_PROMPT = "highly detailed, 8k, professional photography"
NEGATIVE_PROMPT = "blurry, low quality, distorted, ugly"
WIDTH, HEIGHT = 1024, 1024

prompts = [p.strip() for p in PROMPTS_TEXT.split('\n') if p.strip()]
print(f"Generating {len(prompts)} images...")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print("\nLoading SDXL Lightning...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

pipe.load_lora_weights("ByteDance/SDXL-Lightning", 
                       weight_name="sdxl_lightning_8step_lora.safetensors")

pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing")
pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

print("Model loaded!\n")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/kaggle/working/images_{ts}"
os.makedirs(out_dir)

paths = []
for i, prompt in enumerate(prompts, 1):
    try:
        clear_mem()
        full_prompt = f"{prompt}, {SUB_PROMPT}" if SUB_PROMPT else prompt
        seed = random.randint(0, 2**32-1)
        
        print(f"[{i}/{len(prompts)}] {prompt[:50]}...")
        
        img = pipe(
            prompt=full_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=8,
            guidance_scale=1.0,
            width=WIDTH,
            height=HEIGHT,
            generator=torch.manual_seed(seed)
        ).images[0]
        
        filename = f"img_{i:04d}_{seed}.png"
        filepath = os.path.join(out_dir, filename)
        img.save(filepath)
        paths.append(filepath)
        print(f"Saved {filename}")
        
        clear_mem()
    except Exception as e:
        print(f"Error: {e}")

zip_file = f"/kaggle/working/images_{ts}.zip"
with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        z.write(p, os.path.basename(p))

print(f"\n{'='*70}")
print(f"Complete! {len(paths)}/{len(prompts)} images")
print(f"ZIP: {zip_file}")
print(f"{'='*70}")
