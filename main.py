import os
import sys
import time
import threading
import runpod
from server.request_queue import VideoRequest, Job
from server.utils import webhook_response, delete_old_files
from server.request_processor import process_job
from server import server_settings

runpod.api_key = server_settings.RUNPOD_API_KEY


# ============================================================
# MODEL PRE-DOWNLOAD
#
# Downloads the Wan 2.2 model to cache at startup (CPU-only).
# This runs BEFORE runpod.serverless.start() so the model is
# ready when the first request arrives.
#
# Uses same cache_dir as from_pretrained() for compatibility.
# Sets HF_HUB_OFFLINE=1 after download to prevent network calls.
# ============================================================

def ensure_model_cached():
    """
    Ensure the Wan 2.2 model files are present in MODEL_CACHE_DIR.

    Both snapshot_download() and from_pretrained() use the same HF cache
    layout when given the same cache_dir, so files are shared correctly.
    """
    cache_dir = server_settings.MODEL_CACHE_DIR
    hf_home = os.environ.get("HF_HOME", "/app/huggingface")
    model_id = server_settings.I2V_MODEL_ID
    token = os.environ.get("HF_TOKEN") or None

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(hf_home, exist_ok=True)

    print("=" * 60)
    print(f"Checking model cache: {model_id}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  HF home:   {hf_home}")
    print("=" * 60)

    try:
        from huggingface_hub import snapshot_download, try_to_load_from_cache
        from huggingface_hub.utils import LocalEntryNotFoundError

        # Quick check: is the model already fully cached?
        try:
            cached_path = try_to_load_from_cache(
                model_id,
                filename="model_index.json",
                cache_dir=cache_dir,
            )
            if cached_path is not None:
                print(f"Model already cached at {cache_dir}")
                # Set offline mode to prevent any network calls during loading
                os.environ["HF_HUB_OFFLINE"] = "1"
                print("  HF_HUB_OFFLINE=1 (no network calls during model load)")
                print("=" * 60)
                return True
        except (LocalEntryNotFoundError, Exception):
            pass

        # Model not cached — download it
        print("Model NOT found in cache. Starting download...")
        print("  This downloads ~126GB on first boot.")
        print("-" * 60)

        dl_start = time.time()
        snapshot_download(
            model_id,
            cache_dir=cache_dir,
            token=token,
            resume_download=True,
        )
        dl_time = time.time() - dl_start
        print("-" * 60)
        print(f"Model download complete: {dl_time:.0f}s ({dl_time/60:.1f} min)")

        # Set offline mode after successful download
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("  HF_HUB_OFFLINE=1 (no network calls during model load)")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"WARNING: Model pre-download failed: {e}")
        print("  The model will attempt to download on first request instead.")
        print("  Check: network connectivity, disk space, HF_TOKEN if gated")
        print("=" * 60)
        return False


# ============================================================
# LAZY GPU PIPELINE LOADING (thread-safe singleton)
#
# Uses a threading.Lock to prevent double-initialization if
# RunPod sends concurrent requests (which it can with max_concurrency > 1).
# ============================================================
_video_service = None
_video_service_lock = threading.Lock()


def get_video_service():
    """Lazy-load VideoService on first request. Thread-safe."""
    global _video_service
    if _video_service is not None:
        return _video_service

    with _video_service_lock:
        # Double-check after acquiring lock
        if _video_service is not None:
            return _video_service

        import torch
        from server.video_service import VideoService

        print("=" * 60)
        print(f"Initializing VideoService — Wan {server_settings.WAN_VERSION}")
        print(f"  I2V Model: {server_settings.I2V_MODEL_ID}")
        print(f"  Cache Dir: {server_settings.MODEL_CACHE_DIR}")
        print(f"  VAE Tiling: DISABLED (broken for Wan 2.2)")
        print(f"  CPU Offload: {server_settings.ENABLE_CPU_OFFLOAD}")
        print(f"  SageAttention: {server_settings.ENABLE_SAGE_ATTENTION}")
        print(f"  CacheDiT: {server_settings.ENABLE_TEACACHE}")
        print(f"  Post-Processing: {server_settings.ENABLE_POST_PROCESSING}")
        if server_settings.ENABLE_POST_PROCESSING:
            print(f"    Face Restore: {server_settings.ENABLE_FACE_RESTORE}")
            print(f"    Interpolation: {server_settings.ENABLE_INTERPOLATION} → {server_settings.TARGET_FPS}fps")
            print(f"    Upscale: {server_settings.ENABLE_UPSCALE} ({server_settings.UPSCALE_FACTOR}x)")
            print(f"    FFmpeg Enhance: {server_settings.ENABLE_FFMPEG_ENHANCE}")
        print("=" * 60)

        _video_service = VideoService()

        gpu_info = _video_service.get_gpu_info()
        print("=" * 60)
        print(f"VideoService ready. GPU: {gpu_info.get('gpu', 'N/A')}")
        print(f"  VRAM Total: {gpu_info.get('vram_total_gb', 0)}GB")
        print(f"  VRAM Used: {gpu_info.get('vram_used_gb', 0)}GB")
        print(f"  SageAttention: {gpu_info.get('sage_attention', False)}")
        print(f"  CacheDiT: {gpu_info.get('teacache', False)}")
        print(f"  Post-Processing: {gpu_info.get('post_processing', False)}")
        if gpu_info.get('post_processing_info'):
            pp = gpu_info['post_processing_info']
            print(f"    CodeFormer: {'✓' if pp.get('codeformer_available') else '✗'}")
            print(f"    GFPGAN: {'✓' if pp.get('gfpgan_available') else '✗'}")
            print(f"    Real-ESRGAN: {'✓' if pp.get('realesrgan_available') else '✗'}")
            print(f"    RIFE: {'✓' if pp.get('rife_available') else '✗ (FFmpeg fallback)'}")
        print("=" * 60)

        # Warmup CUDA kernels
        if torch.cuda.is_available():
            try:
                print("Running CUDA warmup...")
                dummy = torch.randn(1, 4, 1, 32, 32, device="cuda", dtype=torch.bfloat16)
                _ = dummy * 2
                del dummy, _
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print("CUDA warmup complete.")
            except Exception as e:
                print(f"CUDA warmup failed (non-fatal): {e}")

    return _video_service


def process_request_payload(request_dict):
    """Validate and parse incoming request payload."""
    job_id = request_dict.get("job_id")
    job_request_params = request_dict.get("job_request_params", None)
    webhook_url = request_dict.get("webhook_url", None)

    # Validate webhook_url
    if not webhook_url or not isinstance(webhook_url, str) or "http" not in webhook_url:
        print("No valid webhook url provided!")
        return None

    # Validate job_id
    if not job_id or not isinstance(job_id, str):
        print("No valid job id provided!")
        webhook_response(
            webhook_url, False, 400, "No valid job id provided!", {"job_id": None}
        )
        return None

    # Validate job_request_params — accept both list and single dict
    if isinstance(job_request_params, dict):
        # Single request passed as dict instead of list — wrap it
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

    # Validate each video request
    job = Job(job_id=job_id, webhook_url=webhook_url)
    for idx, req in enumerate(job_request_params):
        if not isinstance(req, dict):
            print(f"Invalid request format at index {idx}")
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
            print(f"No valid prompt provided at index {idx}")
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
            print(f"Invalid request params at index {idx}: {e}")
            webhook_response(
                webhook_url,
                False,
                400,
                f"Invalid request params: {str(e)}",
                {"job_id": job_id},
            )
            return None

    return job


async def callback(data):
    """Process a single incoming request payload."""
    if not data:
        print("No request payload found!")
        return {"error": "No request payload found!"}

    print(f"\nReceived job request: {data.get('job_id', 'unknown')}")
    job = process_request_payload(data)

    if not job:
        print("Failed to create job from payload!")
        return {"error": "Failed to create job from payload!"}

    try:
        video_service = get_video_service()
        await process_job(job, video_service)
        try:
            delete_old_files(server_settings.TEMP_DIR)
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
        return {"status": "success"}
    except Exception as e:
        print(f"Error processing message: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def receive_job(event):
    """RunPod serverless handler entry point."""
    data_for_job = event["input"]
    if not data_for_job:
        return {"error": "No JSON data provided"}
    await callback(data_for_job)
    return "Video generated successfully"


if __name__ == "__main__":
    print("=" * 60)
    print("Starting RunPod handler — Wan 2.2 + Post-Processing Pipeline")
    print(f"  CUDA alloc config: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')}")
    print("=" * 60)

    # Step 1: Ensure model files are downloaded (CPU-only, no GPU needed)
    ensure_model_cached()

    # Step 2: Start the RunPod serverless handler
    # GPU pipeline loads lazily on first request via get_video_service()
    print("Starting RunPod serverless handler...")
    print("  GPU pipeline loads on first request (~1-3 min)")
    print("  IMPORTANT: Set endpoint execution_timeout to 1800000 (30 min)")
    print("=" * 60)
    runpod.serverless.start({"handler": receive_job})
