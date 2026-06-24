# Stillus RealVisXL worker with IP-Adapter FaceID
import os
import base64
import requests
from io import BytesIO
from PIL import Image

import cv2
import numpy as np
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
)

from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from schemas import INPUT_SCHEMA

torch.cuda.empty_cache()

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[Model] insightface not available — face consistency disabled")


class ModelHandler:
    def __init__(self):
        self.base = None
        self.face_app = None
        self.ip_adapter_loaded = False
        self.load_models()

    def load_base(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0",
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")

        try:
            base_pipe.load_ip_adapter(
                "h94/IP-Adapter-FaceID",
                subfolder=None,
                weight_name="ip-adapter-faceid_sdxl.bin",
                local_files_only=True,
            )
            base_pipe.set_ip_adapter_scale(0.6)
            self.ip_adapter_loaded = True
            print("[Model] IP-Adapter FaceID loaded")
        except Exception as e:
            print(f"[Model] IP-Adapter FaceID failed: {e}")
            self.ip_adapter_loaded = False

        base_pipe.enable_xformers_memory_efficient_attention()
        return base_pipe

    def load_face_analysis(self):
        if not INSIGHTFACE_AVAILABLE:
            return None
        try:
            app = FaceAnalysis(
                name="buffalo_l",
                root="/models/insightface",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            print("[Model] InsightFace loaded")
            return app
        except Exception as e:
            print(f"[Model] InsightFace failed: {e}")
            return None

    def load_models(self):
        self.base = self.load_base()
        self.face_app = self.load_face_analysis()

    def get_face_embedding(self, image_url):
        if not self.face_app:
            return None
        try:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            faces = self.face_app.get(img_array)
            if not faces:
                print("[FaceID] No face detected in reference image")
                return None
            emb = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
            print("[FaceID] Face embedding extracted")
            return emb
        except Exception as e:
            print(f"[FaceID] Failed: {e}")
            return None


MODELS = ModelHandler()


def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)
        if os.environ.get("BUCKET_ENDPOINT_URL", False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")
    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
        "DPMSolverSinglestep": DPMSolverSinglestepScheduler.from_config(config),
    }[name]


@torch.inference_mode()
def generate_image(job):
    job_input = job["input"]

    # Extract face_reference_url before validation (not in INPUT_SCHEMA)
    face_ref_url = job_input.pop("face_reference_url", None)

    validated_input = validate(job_input, INPUT_SCHEMA)
    if "errors" in validated_input:
        return {"error": validated_input["errors"]}
    job_input = validated_input["validated_input"]

    if job_input["seed"] is None:
        job_input["seed"] = int.from_bytes(os.urandom(2), "big")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(job_input["seed"])

    MODELS.base.scheduler = make_scheduler(
        job_input["scheduler"], MODELS.base.scheduler.config
    )

    # Extract face embedding if reference provided
    face_emb = None
    if face_ref_url and MODELS.ip_adapter_loaded:
        face_emb = MODELS.get_face_embedding(face_ref_url)
        if face_emb is not None:
            MODELS.base.set_ip_adapter_scale(0.6)
        else:
            MODELS.base.set_ip_adapter_scale(0.0)
    elif MODELS.ip_adapter_loaded:
        MODELS.base.set_ip_adapter_scale(0.0)

    try:
        with torch.inference_mode():
            gen_kwargs = dict(
                prompt=job_input["prompt"],
                negative_prompt=job_input["negative_prompt"],
                height=job_input["height"],
                width=job_input["width"],
                num_inference_steps=job_input["num_inference_steps"],
                guidance_scale=job_input["guidance_scale"],
                num_images_per_prompt=job_input["num_images"],
                generator=generator,
            )
            if face_emb is not None:
                gen_kwargs["ip_adapter_image_embeds"] = [face_emb]

            result = MODELS.base(**gen_kwargs)
            output = result.images
    except RuntimeError as err:
        return {"error": f"RuntimeError: {err}", "refresh_worker": True}
    except Exception as err:
        return {"error": f"Unexpected error: {err}", "refresh_worker": True}

    image_urls = _save_and_upload_images(output, job["id"])

    return {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input["seed"],
    }


runpod.serverless.start({"handler": generate_image})
