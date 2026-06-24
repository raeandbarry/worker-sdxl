#!/usr/bin/env python3
"""Download and cache all model weights during Docker build."""

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from huggingface_hub import hf_hub_download
import os

print("[Download] Downloading RealVisXL V4.0...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    add_watermarker=False,
)
print("[Download] ✓ RealVisXL V4.0 cached")

# Download IP-Adapter FaceID weights
print("[Download] Downloading IP-Adapter FaceID for SDXL...")
try:
    hf_hub_download(
        repo_id="h94/IP-Adapter-FaceID",
        filename="ip-adapter-faceid_sdxl.bin",
        local_dir_links=False,
    )
    print("[Download] ✓ IP-Adapter FaceID cached")
except Exception as e:
    print(f"[Download] IP-Adapter FaceID download failed: {e}")
    print("[Download] Face consistency will be disabled — text description only")

# Download InsightFace model for face embedding extraction
print("[Download] Downloading InsightFace buffalo_l model...")
try:
    os.makedirs("/models/insightface/models", exist_ok=True)
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        root="/models/insightface",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("[Download] ✓ InsightFace buffalo_l cached")
except Exception as e:
    print(f"[Download] InsightFace download failed: {e}")
    print("[Download] Face embedding extraction will be disabled")

print("[Download] All downloads complete")
