import os
import sys
import runpod
from server.request_queue import VideoRequest, Job
from server.utils import webhook_response, delete_old_files
from server.request_processor import process_job
from server.video_service import VideoService
from server import server_settings

runpod.api_key = server_settings.RUNPOD_API_KEY

# ============================================================
# GLOBAL MODEL LOADING
#
# Wan 2.2 A14B MoE (Mixture of Experts) pipeline loaded ONCE
# at startup. This runs before RunPod begins accepting jobs.
#
# Model architecture:
#   - High-noise expert: Handles initial layout and composition
#   - Low-noise expert: Refines details, textures, and motion
#   - MoE routing: Automatically selects expert per timestep
#
# Loading takes ~3-5 minutes (first run downloads ~30GB).
# Models are cached at /workspace/models (RunPod network volume).
# No safety filters are applied.
# ============================================================
print("=" * 60)
print(f"Initializing VideoService — Wan {server_settings.WAN_VERSION}")
print(f"  I2V Model: {server_settings.I2V_MODEL_ID}")
print(f"  Cache Dir: {server_settings.MODEL_CACHE_DIR}")
print(f"  VAE Tiling: {server_settings.ENABLE_VAE_TILING}")
print(f"  CPU Offload: {server_settings.ENABLE_CPU_OFFLOAD}")
print(f"  SageAttention: {server_settings.ENABLE_SAGE_ATTENTION}")
print("=" * 60)

video_service = VideoService()

gpu_info = video_service.get_gpu_info()
print("=" * 60)
print(f"VideoService ready. GPU: {gpu_info.get('gpu', 'N/A')}")
print(f"  VRAM Total: {gpu_info.get('vram_total_gb', 0)}GB")
print(f"  VRAM Used: {gpu_info.get('vram_used_gb', 0)}GB")
print(f"  SageAttention: {gpu_info.get('sage_attention', False)}")
print("Starting RunPod handler.")
print("=" * 60)


def process_request_payload(request_dict):
    """
    Validate incoming payload and construct a Job.

    Expected payload for Image-to-Video (Wan 2.2):
    {
        "job_id": "abc-123",
        "webhook_url": "https://api.example.com/webhook",
        "job_request_params": [
            {
                "prompt": "The person is dancing gracefully at a party",
                "reference_image_url": "https://storage.com/person_photo.png",
                "height": 720,
                "width": 1280,
                "num_frames": 81,
                "guidance_scale": 5.0,
                "num_inference_steps": 50,
                "fps": 16,

                // NEW in Wan 2.2:
                "camera_motion": "dolly_in",        // optional
                "enhance_prompt": true,              // auto cinematic enhancement
                "quality_preset": "high"             // draft|standard|high|ultra
            }
        ]
    }

    Camera motion options:
        pan_left, pan_right, pan_up, pan_down,
        zoom_in, zoom_out, static,
        orbit_left, orbit_right,
        dolly_in, dolly_out,
        crane_up, crane_down, handheld

    Quality presets:
        draft    — 30 steps, fast preview
        standard — 50 steps, balanced
        high     — 80 steps, best quality
        ultra    — 100 steps, maximum quality
    """
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
        webhook_response(webhook_url, False, 400, "No valid job id provided!", {"job_id": None})
        return None

    # Validate job_request_params
    if not job_request_params or not isinstance(job_request_params, list) or len(job_request_params) == 0:
        print("No valid job request params provided!")
        webhook_response(webhook_url, False, 400, "No valid job request params provided!", {"job_id": job_id})
        return None

    # Validate each video request
    job = Job(job_id=job_id, webhook_url=webhook_url)
    for idx, req in enumerate(job_request_params):
        if not isinstance(req, dict):
            print(f"Invalid request format at index {idx}")
            webhook_response(webhook_url, False, 400, f"Invalid request format at index {idx}", {"job_id": job_id})
            return None

        # Validate prompt exists and is not empty
        prompt = req.get("prompt")
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            print(f"No valid prompt provided at index {idx}")
            webhook_response(webhook_url, False, 400, f"No valid prompt provided at index {idx}", {"job_id": job_id})
            return None

        try:
            video_request = VideoRequest(**req)
            job.job_request_params.append(video_request)
        except Exception as e:
            print(f"Invalid request params at index {idx}: {e}")
            webhook_response(webhook_url, False, 400, f"Invalid request params: {str(e)}", {"job_id": job_id})
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
        await process_job(job, video_service)
        try:
            delete_old_files(server_settings.TEMP_DIR)
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
        return {"status": "success"}
    except Exception as e:
        print(f"Error processing message: {e}")
        return {"error": str(e)}


async def receive_job(event):
    """RunPod serverless handler entry point."""
    data_for_job = event["input"]
    if not data_for_job:
        return {"error": "No JSON data provided"}
    await callback(data_for_job)
    return "Video generated successfully"


if __name__ == "__main__":
    runpod.serverless.start({"handler": receive_job})
