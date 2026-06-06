import modal
import os

# ---------------------------------------------------------------------------
# 1) Persistent volume for the Wan 2.2 model (~126GB)
# ---------------------------------------------------------------------------

models_volume = modal.Volume.from_name("wan22-models", create_if_missing=True)
MODELS_PATH = "/models"

# ---------------------------------------------------------------------------
# 2) Container image — replicates the Dockerfile
#    - CUDA 12.6 base, Python 3.11, ffmpeg + opencv deps
#    - All training/inference deps, SageAttention, CacheDiT
#    - Post-processing models (CodeFormer, GFPGAN, Real-ESRGAN, facelib) baked in
#    - RIFE cloned + patched basicsr
# ---------------------------------------------------------------------------

video_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "wget",
        "curl",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "torchaudio>=2.5.1",
        "diffusers>=0.36.0,<0.38.0",
        "transformers>=4.49.0",
        "accelerate>=1.3.0",
        "sentencepiece",
        "huggingface_hub>=0.25.0",
        "imageio[ffmpeg]",
        "imageio-ffmpeg",
        "ffmpeg-python",
        "Pillow>=10.0.0",
        "opencv-python-headless",
        "numpy",
        "boto3",
        "pydantic>=2.0",
        "requests",
        "gfpgan>=1.3.8",
        "realesrgan>=0.3.0",
        "lpips",
        "fastapi[standard]",
        "ftfy",
    )
    # Performance optimizations (non-fatal if they fail)
    .run_commands(
        "pip install --no-cache-dir sageattention || echo 'SageAttention install failed, using default attention'",
        "pip install --no-cache-dir cache-dit || echo 'CacheDiT install failed, running without cache acceleration'",
    )
    # Patch basicsr for torchvision 0.20+ and install CodeFormer arch files
    .run_commands(
        "BASICSR_ROOT=$(python3 -c \"import importlib.util; spec = importlib.util.find_spec('basicsr'); print(spec.submodule_search_locations[0])\") && "
        'DEGRADATIONS="${BASICSR_ROOT}/data/degradations.py" && '
        'if [ -f "$DEGRADATIONS" ]; then sed -i \'s/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/\' "$DEGRADATIONS"; fi && '
        'BASICSR_ARCHS="${BASICSR_ROOT}/archs" && '
        'wget -q -O "${BASICSR_ARCHS}/codeformer_arch.py" https://raw.githubusercontent.com/sczhou/CodeFormer/master/basicsr/archs/codeformer_arch.py && '
        'wget -q -O "${BASICSR_ARCHS}/vqgan_arch.py" https://raw.githubusercontent.com/sczhou/CodeFormer/master/basicsr/archs/vqgan_arch.py && '
        "python3 -c \"from basicsr.archs.codeformer_arch import CodeFormer; print('CodeFormer arch verified.')\""
    )
    # Bake post-processing models (~2GB)
    .run_commands(
        "mkdir -p /app/models/codeformer /app/models/gfpgan /app/models/realesrgan /app/models/facelib",
        "wget -q -O /app/models/codeformer/codeformer.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "wget -q -O /app/models/gfpgan/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "wget -q -O /app/models/realesrgan/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "wget -q -O /app/models/facelib/detection_Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth || echo 'facelib detection download failed (auto-downloads at runtime)'",
        "wget -q -O /app/models/facelib/parsing_parsenet.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/parsing_parsenet.pth || echo 'facelib parsing download failed (auto-downloads at runtime)'",
    )
    # RIFE frame interpolation
    .run_commands(
        "git clone https://github.com/hzwer/Practical-RIFE.git /app/rife && cd /app/rife && (git checkout v4.22 2>/dev/null || git checkout main)",
        "mkdir -p /app/rife/train_log",
    )
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.8",
            "GFPGAN_MODEL_PATH": "/app/models/gfpgan/GFPGANv1.4.pth",
            "FACELIB_MODELDIR": "/app/models/facelib",
            "POST_PROCESSING_MODEL_DIR": "/app/models",
        }
    )
    .run_commands("mkdir -p /app/temp /app/control_inputs")
    .add_local_dir("server", "/app/server")
)

# ---------------------------------------------------------------------------
# 3) Modal App
# ---------------------------------------------------------------------------

app = modal.App(
    name="wan22-video-gen",
    image=video_image,
    secrets=[
        modal.Secret.from_name("z-image-turbo-secrets"),
    ],
)

# ---------------------------------------------------------------------------
# 4) One-time model download — Wan 2.2 (~126GB) into the volume
#    modal run app.py::download_models
# ---------------------------------------------------------------------------


@app.function(
    volumes={MODELS_PATH: models_volume},
    timeout=14400,  # 4 hours — 126GB download
)
def download_models():
    import os
    from huggingface_hub import snapshot_download

    model_id = os.environ.get("I2V_MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    token = os.environ.get("HF_TOKEN") or None

    print(f"Downloading {model_id} into volume at {MODELS_PATH} ...")
    print("This is ~126GB and may take 30-60 min on first run.")
    snapshot_download(
        model_id,
        cache_dir=MODELS_PATH,
        token=token,
        resume_download=True,
    )
    models_volume.commit()
    print("Download complete and volume committed.")


# ---------------------------------------------------------------------------
# 5) GPU generation function — runs in the BACKGROUND on H200
# ---------------------------------------------------------------------------


@app.function(
    gpu="H200",  # 141GB — fits full A14B without CPU offload
    timeout=3600,  # 1 hour max per video job
    scaledown_window=60,
    volumes={MODELS_PATH: models_volume},
    startup_timeout=1800,  # 30 min — model load is heavy
)
def run_generation(data: dict):
    """
    Background video generation. Spawned by the web endpoint.
    Runs Wan 2.2 I2V + 4-stage post-processing. Webhook fires on completion.
    """
    import os, sys, asyncio, traceback

    os.chdir("/app")
    sys.path.insert(0, "/app")

    os.environ["MODEL_CACHE_DIR"] = MODELS_PATH
    os.environ["HF_HOME"] = os.path.join(MODELS_PATH, "huggingface")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    from server import server_settings
    from server.request_queue import VideoRequest, Job
    from server.utils import webhook_response, delete_old_files
    from server.request_processor import process_job

    # Validate + build job
    job = _build_job(data)
    if not job:
        print("Failed to build job from payload.")
        return

    try:
        # Lazy-load the video service (loads Wan 2.2 into VRAM)
        from server.video_service import VideoService

        print("Initializing VideoService (Wan 2.2 I2V A14B)...")
        video_service = VideoService()

        gpu_info = video_service.get_gpu_info()
        print(
            f"GPU: {gpu_info.get('gpu')}, VRAM total: {gpu_info.get('vram_total_gb')}GB"
        )

        # Run generation (async)
        asyncio.run(process_job(job, video_service))

        try:
            delete_old_files(server_settings.TEMP_DIR)
        except Exception as e:
            print(f"Cleanup error (non-fatal): {e}")

        print("Generation job finished.")

    except Exception as e:
        print(f"Error processing video job: {e}")
        traceback.print_exc()
        webhook_response(
            job.webhook_url,
            False,
            500,
            f"Video generation failed: {str(e)}",
            {"job_id": job.job_id},
        )


def _build_job(data: dict):
    """Validate and parse incoming payload into a Job."""
    import sys

    sys.path.insert(0, "/app")
    from server.request_queue import VideoRequest, Job
    from server.utils import webhook_response

    job_id = data.get("job_id")
    job_request_params = data.get("job_request_params", None)
    webhook_url = data.get("webhook_url", None)

    if not webhook_url or not isinstance(webhook_url, str) or "http" not in webhook_url:
        print("No valid webhook url provided!")
        return None

    if not job_id or not isinstance(job_id, str):
        print("No valid job id provided!")
        webhook_response(
            webhook_url, False, 400, "No valid job id provided!", {"job_id": None}
        )
        return None

    if isinstance(job_request_params, dict):
        job_request_params = [job_request_params]

    if (
        not job_request_params
        or not isinstance(job_request_params, list)
        or len(job_request_params) == 0
    ):
        print("No valid job request params provided!")
        webhook_response(
            webhook_url,
            False,
            400,
            "No valid job request params provided!",
            {"job_id": job_id},
        )
        return None

    job = Job(job_id=job_id, webhook_url=webhook_url)
    for idx, req in enumerate(job_request_params):
        if not isinstance(req, dict):
            webhook_response(
                webhook_url,
                False,
                400,
                f"Invalid request format at index {idx}",
                {"job_id": job_id},
            )
            return None
        prompt = req.get("prompt")
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            webhook_response(
                webhook_url,
                False,
                400,
                f"No valid prompt provided at index {idx}",
                {"job_id": job_id},
            )
            return None
        try:
            video_request = VideoRequest(**req)
            job.job_request_params.append(video_request)
        except Exception as e:
            webhook_response(
                webhook_url,
                False,
                400,
                f"Invalid request params: {str(e)}",
                {"job_id": job_id},
            )
            return None

    return job


# ---------------------------------------------------------------------------
# 6) Lightweight web endpoint — NO GPU, returns immediately
# ---------------------------------------------------------------------------


@app.function(timeout=60)
@modal.fastapi_endpoint(method="POST", docs=True)
def generate(data: dict) -> dict:
    """
    HTTP endpoint for video generation.
    - Normal: validates input, spawns background GPU job, returns job_id + call_id
    - Cancel: POST {"action": "cancel", "call_id": "..."} to cancel a running job
    """
    if not data:
        return {"error": "No JSON data provided"}

    # ----- Cancel mode -----
    if data.get("action") == "cancel":
        call_id = data.get("call_id")
        if not call_id:
            return {"error": "No call_id provided"}
        try:
            modal.FunctionCall.from_id(call_id).cancel()
            return {"status": "cancelled", "call_id": call_id}
        except Exception as e:
            return {"error": f"Failed to cancel: {str(e)}"}

    # ----- Normal generation mode -----
    job_id = data.get("job_id", "")
    webhook_url = data.get("webhook_url")
    job_request_params = data.get("job_request_params")

    if not webhook_url:
        return {"error": "No webhook_url provided"}
    if isinstance(job_request_params, dict):
        job_request_params = [job_request_params]
    if (
        not job_request_params
        or not isinstance(job_request_params, list)
        or len(job_request_params) == 0
    ):
        return {"error": "No valid job_request_params provided"}

    function_call = run_generation.spawn(data)

    return {
        "status": "accepted",
        "job_id": job_id,
        "call_id": function_call.object_id,
        "message": "Video job accepted. Webhook will fire on completion.",
    }


# ---------------------------------------------------------------------------
# 7) Local test entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def test():
    print("Wan 2.2 Video Gen deployed on Modal.")
    print("One-time: modal run app.py::download_models")
    print("Then POST to the generate endpoint.")
