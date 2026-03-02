# VideoGen-RunPod — Wan 2.2 A14B MoE

Serverless video generation on RunPod using **Wan 2.2 A14B** with Mixture-of-Experts architecture.

## What Changed from Wan 2.1

| Feature | Wan 2.1 | Wan 2.2 |
|---|---|---|
| Architecture | Single model | MoE (high-noise + low-noise experts) |
| Training data | Baseline | +65% images, +83% videos |
| Motion quality | Good | Cinematic (gestures, athletics, expressions) |
| Semantic control | Basic | Multi-object, spatial accuracy |
| Aesthetic control | Limited | Lighting, color, composition via prompt |
| Camera control | Via prompt only | Dedicated camera_motion parameter |
| Quality presets | None | draft / standard / high / ultra |
| Prompt enhancement | None | Auto-cinematic prompt enrichment |
| Video encoding | Basic MP4 | H.264 CRF 18 high-quality |
| Attention | Default SDPA | SageAttention (40-50% faster) |
| FPS default | 15 | 16 (Wan 2.2 native) |

## GPU Requirements

| GPU | VRAM | Mode | Speed (81 frames, 480p, 50 steps) |
|---|---|---|---|
| H100 80GB | 80GB | Full VRAM | ~2-3 min |
| A100 80GB | 80GB | Full VRAM | ~3-5 min |
| L40S 48GB | 48GB | CPU offload | ~6-10 min |
| A6000 48GB | 48GB | CPU offload | ~8-12 min |

## API Payload

### Image-to-Video (primary mode)

```json
{
    "job_id": "abc-123",
    "webhook_url": "https://api.example.com/webhook",
    "job_request_params": [
        {
            "prompt": "The woman walks confidently through a modern city at golden hour",
            "reference_image_url": "https://storage.com/person.png",
            "height": 720,
            "width": 1280,
            "num_frames": 81,
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
            "fps": 16,
            "camera_motion": "dolly_in",
            "enhance_prompt": true,
            "quality_preset": "high"
        }
    ]
}
```

### New Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `camera_motion` | string \| null | null | Camera movement control |
| `enhance_prompt` | bool | true | Auto-enhance prompt for cinematic quality |
| `quality_preset` | string \| null | null | Quick quality override |

### Camera Motion Options

| Value | Effect |
|---|---|
| `pan_left` / `pan_right` | Horizontal camera pan |
| `pan_up` / `pan_down` | Vertical camera tilt |
| `zoom_in` / `zoom_out` | Camera zoom |
| `dolly_in` / `dolly_out` | Camera dolly movement |
| `orbit_left` / `orbit_right` | Orbital camera rotation |
| `crane_up` / `crane_down` | Crane shot |
| `static` | Locked tripod shot |
| `handheld` | Natural handheld shake |

### Quality Presets

| Preset | Steps | Guidance | Use Case |
|---|---|---|---|
| `draft` | 25 | 3.5 | Fast preview |
| `standard` | 40 | 3.5 | Balanced quality/speed |
| `high` | 50 | 4.0 | Best quality |
| `ultra` | 80 | 4.5 | Maximum quality |

## Prompt Tips for Wan 2.2

Wan 2.2 follows this prompt structure for best results:

```
Subject (description) + Scene (description) + Motion (description) + Aesthetic Control + Style
```

**Examples:**

```
# Good — structured, specific
"A young woman with flowing dark hair, wearing a red silk dress, walks through a sunlit garden with blooming roses, gentle breeze moving her hair, soft golden hour lighting, anamorphic bokeh, cinematic color grading"

# Better — with motion and aesthetic detail
"Close-up of a man's face as he slowly turns toward camera, subtle smile forming, dramatic Rembrandt lighting from the left, shallow depth of field, film grain, warm teal-and-orange color grade, 35mm lens"
```

**Aesthetic tags that work well with Wan 2.2:**
- Lighting: `volumetric dusk`, `harsh noon sun`, `neon rim light`, `golden hour`
- Color: `teal-and-orange`, `bleach-bypass`, `kodak portra`, `warm color grade`
- Lens: `anamorphic bokeh`, `16mm grain`, `35mm lens`, `shallow depth of field`
- Motion: `slow-motion`, `whip-pan`, `time-lapse`

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `I2V_MODEL_ID` | `Wan-AI/Wan2.2-I2V-A14B` | HuggingFace model ID |
| `T2V_MODEL_ID` | `Wan-AI/Wan2.2-T2V-A14B` | HuggingFace model ID |
| `MODEL_CACHE_DIR` | auto-detected | Model cache path (auto-detects `/runpod-volume`) |
| `ENABLE_CPU_OFFLOAD` | `true` | Model CPU offload (required for A14B) |
| `ENABLE_SEQUENTIAL_CPU_OFFLOAD` | `false` | Sequential offload (lowest VRAM) |
| `ENABLE_SAGE_ATTENTION` | `true` | SageAttention for faster inference |
| `ENABLE_TEACACHE` | `true` | CacheDiT acceleration (~2-3x speedup) |
| `ENABLE_TORCH_COMPILE` | `false` | torch.compile (incompatible with Wan 2.2) |
| `HF_TOKEN` | — | HuggingFace token (optional, avoids rate limits) |
| `RUNPOD_API_KEY` | — | RunPod API key (required) |
| `DIGITAL_OCEAN_*` | — | Storage credentials (required) |

## Deployment on RunPod

1. Build and push the Docker image
2. Create a RunPod serverless endpoint with the image
3. Attach a network volume at `/runpod-volume` (models persist here)
4. Set environment variables
5. First boot downloads Wan 2.2 A14B (~126GB) — subsequent boots use cache

For 48GB GPUs (L40S/A6000), set `ENABLE_CPU_OFFLOAD=true`.
