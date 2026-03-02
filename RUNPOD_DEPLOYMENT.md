# RunPod Serverless Deployment — Wan 2.2 A14B Video Generation

## Architecture Overview

The Docker image ships at **~15-20GB** with all code, dependencies, and post-processing models.
The main Wan 2.2 model (**~126GB**) downloads at runtime to a **Network Volume**, so:

- Builds complete in **5-10 minutes** (no disk space issues)
- First boot downloads the model once (~10-20 min on RunPod)
- Subsequent boots load from the volume instantly
- No need for 300GB+ free disk during build

## Prerequisites

- Docker with BuildKit enabled
- RunPod account with **Network Volume** (200GB recommended)
- A100 80GB or H100 GPU (required for A14B model)
- Optional: HuggingFace token (Wan 2.2 is public, but token avoids rate limits)

## Step 1: Build the Docker Image

```bash
# Clean up any previous build artifacts
docker builder prune -a --force
docker system prune -a --volumes --force

# Build (fast — no model download during build)
DOCKER_BUILDKIT=1 docker build -t your-registry/videogen-wan:v3 .

# Push to your registry (Docker Hub, GHCR, etc.)
docker push your-registry/videogen-wan:v3
```

**Build time:** ~5-10 minutes
**Image size:** ~15-20GB

## Step 2: Create a RunPod Network Volume

1. Go to **RunPod Console** → **Storage** → **Network Volumes**
2. Create a volume:
   - **Name:** `wan-models`
   - **Size:** 200GB (126GB model + overhead)
   - **Region:** Same as your serverless endpoint

## Step 3: Create Serverless Endpoint

1. Go to **Serverless** → **New Endpoint**
2. Configure:
   - **Container Image:** `your-registry/videogen-wan:v3`
   - **GPU:** A100 80GB (or H100)
   - **Network Volume:** Select `wan-models`, mount at `/runpod-volume`
   - **Container Disk:** 20GB (just for temp files)
   - **Idle Timeout:** 300s (keeps worker warm between requests)
   - **Execution Timeout:** 1800s (30 min — allows for generation + post-processing)

3. Set **Environment Variables:**

| Variable | Value | Required |
|----------|-------|----------|
| `RUNPOD_API_KEY` | Your RunPod API key | Yes |
| `DIGITAL_OCEAN_ENDPOINT_URL` | Your DO Spaces endpoint | Yes |
| `DIGITAL_OCEAN_BUCKET_ACCESS_KEY` | DO access key | Yes |
| `DIGITAL_OCEAN_BUCKET_SECRET_KEY` | DO secret key | Yes |
| `DIGITAL_OCEAN_BUCKET_NAME` | Bucket name | Yes |
| `DIGITAL_OCEAN_BUCKET_URL` | Bucket CDN URL | Yes |
| `HF_TOKEN` | Your HuggingFace token | Optional |
| `MODEL_CACHE_DIR` | `/runpod-volume/models` | Optional |
| `HF_HOME` | `/runpod-volume/huggingface` | Optional |

> **Note:** `MODEL_CACHE_DIR` and `HF_HOME` are auto-detected when `/runpod-volume` is mounted. You only need to set them if using a custom path.

## Step 4: First Boot

On the first cold start with an empty network volume:

1. Worker starts, checks model cache → not found
2. Downloads Wan 2.2 A14B (~126GB) → takes ~10-20 min
3. Starts the RunPod handler
4. First request triggers GPU pipeline initialization (~1-3 min)
5. Video generation begins

**Subsequent cold starts** (model cached on volume):
1. Worker starts, checks model cache → found ✓
2. Starts handler immediately
3. First request loads GPU pipeline (~1-3 min)
4. Video generation begins

## API Usage

### Request Format

```json
{
  "input": {
    "job_id": "unique-job-id",
    "webhook_url": "https://your-server.com/webhook",
    "job_request_params": [
      {
        "prompt": "A woman walking through a garden, gentle breeze",
        "reference_image_url": "https://example.com/photo.jpg",
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "guidance_scale": 3.5,
        "num_inference_steps": 40,
        "enable_post_processing": true,
        "enable_face_restore": true,
        "enable_interpolation": true,
        "enable_upscale": true,
        "enable_ffmpeg_enhance": true,
        "target_fps": 24,
        "upscale_factor": 2.0,
        "face_fidelity": 0.6
      }
    ]
  }
}
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `guidance_scale` | 3.5 | Wan team recommended for I2V (range: 3.0-5.0) |
| `num_inference_steps` | 40 | ~22 effective with CacheDiT |
| `num_frames` | 81 | Must satisfy (n-1) % 4 == 0. Max 161 |
| `quality_preset` | null | `draft`, `standard`, `high`, `ultra` |
| `camera_motion` | null | `pan_left`, `zoom_in`, `orbit_right`, etc. |
| `enable_post_processing` | true | Master toggle for 4-stage pipeline |
| `face_fidelity` | 0.6 | CodeFormer w: 0=quality, 1=identity |

### Webhook Response

```json
{
  "status": true,
  "code": 200,
  "message": "Video Generated!",
  "data": {
    "job_id": "unique-job-id",
    "video_url": "https://your-bucket.cdn.com/videos/2025-01-15/unique-job-id/video.mp4",
    "duration_seconds": 5.06,
    "num_frames": 121,
    "resolution": "1664x960",
    "fps": 24,
    "post_processed": true,
    "model_version": "wan2.2-i2v-a14b"
  }
}
```

## Alternative: Bake Model Into Image

If you prefer baking the model into the image (no network volume needed):

1. Add a `RUN` step in `Dockerfile` to download the model during build (before `CMD`)
2. Ensure **~300GB free disk** during build
3. Build with HF token:
   ```bash
   DOCKER_BUILDKIT=1 docker build \
     --build-arg HF_TOKEN=$HF_TOKEN \
     -t your-registry/videogen-wan:v3-baked .
   ```
4. Image will be ~140-150GB
5. No network volume needed on RunPod

## Troubleshooting

**Model download hangs or fails:**
- Check network connectivity: `curl -I https://huggingface.co`
- Check disk space on network volume: `df -h /runpod-volume`
- Set `HF_TOKEN` env var to avoid rate limits

**Out of VRAM:**
- Ensure A100 80GB is selected (A14B needs ~35GB+ for inference)
- Enable CPU offload: `ENABLE_CPU_OFFLOAD=true`
- Reduce resolution or frame count

**Post-processing crashes:**
- Real-ESRGAN auto-adjusts tile size for VRAM
- Disable stages individually: `ENABLE_UPSCALE=false`
- Check logs for which stage fails

**Worker timeout before model loads:**
- Increase idle timeout to 600s
- Once model is cached on volume, subsequent boots are fast
