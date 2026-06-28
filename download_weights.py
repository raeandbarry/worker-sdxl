"""
Pre-download all model weights during Docker build.
This bakes them into the image so cold starts don't need to fetch anything.
The handler uses local_files_only=True, so everything must be cached here.
"""

import os

# Ensure HuggingFace cache is in a predictable location
os.environ["HF_HOME"] = "/root/.cache/huggingface"

from huggingface_hub import snapshot_download

print("[Weights] Downloading SG161222/RealVisXL_V4.0...")
snapshot_download("SG161222/RealVisXL_V4.0")

print("[Weights] Downloading madebyollin/sdxl-vae-fp16-fix...")
snapshot_download("madebyollin/sdxl-vae-fp16-fix")

print("[Weights] Downloading h94/IP-Adapter-FaceID...")
snapshot_download("h94/IP-Adapter-FaceID")

# InsightFace buffalo_l model
print("[Weights] Downloading InsightFace buffalo_l...")
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        root="/models/insightface",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("[Weights] InsightFace buffalo_l ready")
except Exception as e:
    print(f"[Weights] InsightFace download failed (non-fatal): {e}")

print("[Weights] All models downloaded successfully")
