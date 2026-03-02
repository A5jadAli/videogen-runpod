import os

# ============================================================
# CUDA Memory Configuration (set BEFORE any torch imports)
# expandable_segments prevents fragmentation OOM on long runs
# garbage_collection_threshold proactively reclaims at 80% usage
# ============================================================
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)

# Base directory of project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Temp directory for generated videos and control inputs
TEMP_DIR = os.path.join(BASE_DIR, "temp")
CONTROL_DIR = os.path.join(BASE_DIR, "control_inputs")

# ============================================================
# Model settings — Wan 2.2 A14B MoE Architecture
# ============================================================
WAN_VERSION = os.environ.get("WAN_VERSION", "2.2")

# Wan 2.2 model IDs (Diffusers format)
T2V_MODEL_ID = os.environ.get("T2V_MODEL_ID", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
I2V_MODEL_ID = os.environ.get("I2V_MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")

# ============================================================
# Model Cache Directory — auto-detects RunPod Network Volume
#
# Priority:
#   1. MODEL_CACHE_DIR env var (explicit override)
#   2. /runpod-volume/models (RunPod network volume, if mounted)
#   3. /app/models (default — baked into image or first-boot download)
# ============================================================
def _resolve_model_cache_dir():
    """Determine the best model cache directory."""
    explicit = os.environ.get("MODEL_CACHE_DIR")
    if explicit:
        return explicit

    # Auto-detect RunPod network volume
    runpod_vol = "/runpod-volume"
    if os.path.isdir(runpod_vol) and os.access(runpod_vol, os.W_OK):
        vol_models = os.path.join(runpod_vol, "models")
        os.makedirs(vol_models, exist_ok=True)
        if not os.environ.get("HF_HOME"):
            hf_vol = os.path.join(runpod_vol, "huggingface")
            os.makedirs(hf_vol, exist_ok=True)
            os.environ["HF_HOME"] = hf_vol
        return vol_models

    return "/app/models"


MODEL_CACHE_DIR = _resolve_model_cache_dir()

# ============================================================
# Post-processing model directory
# ============================================================
POST_PROCESSING_MODEL_DIR = os.environ.get("POST_PROCESSING_MODEL_DIR", "/app/models")

# ============================================================
# Memory + Performance optimizations
#
# CRITICAL: CPU offload is REQUIRED for A14B (27B params total).
# Without it, the model needs ~55GB+ VRAM just for weights.
# enable_model_cpu_offload() keeps peak at ~35-50GB on 80GB GPU.
# ============================================================
ENABLE_CPU_OFFLOAD = os.environ.get("ENABLE_CPU_OFFLOAD", "true").lower() == "true"
ENABLE_SEQUENTIAL_CPU_OFFLOAD = os.environ.get("ENABLE_SEQUENTIAL_CPU_OFFLOAD", "false").lower() == "true"

# CRITICAL: VAE tiling is BROKEN for AutoencoderKLWan (diffusers #12529).
# Enabling it causes tensor dimension mismatch crashes. Do NOT enable.
ENABLE_VAE_TILING = os.environ.get("ENABLE_VAE_TILING", "false").lower() == "true"

# VAE slicing only helps for batch>1 generation, harmless otherwise
ENABLE_VAE_SLICING = os.environ.get("ENABLE_VAE_SLICING", "false").lower() == "true"

# SageAttention — ~2x faster attention. MUST use pinned version.
# Known broken: SageAttention 2.2.0 SM90 backend on H100 (#320),
# SageAttention + FP8 models (#221), SageAttention + 5B variant (#8573).
# Safe: SageAttention2 8-bit with BF16 A14B models.
ENABLE_SAGE_ATTENTION = os.environ.get("ENABLE_SAGE_ATTENTION", "true").lower() == "true"

# CacheDiT — TaylorSeer + DBCache ~2-3x inference speedup
# Preferred over TeaCache for MoE architectures (dual-transformer aware)
ENABLE_TEACACHE = os.environ.get("ENABLE_TEACACHE", "true").lower() == "true"

# torch.compile — INCOMPATIBLE with Wan 2.2 in fullgraph mode (#12728)
# Do NOT enable unless you've verified your exact diffusers+torch version.
ENABLE_TORCH_COMPILE = os.environ.get("ENABLE_TORCH_COMPILE", "false").lower() == "true"

# ============================================================
# Quality defaults for Wan 2.2 I2V (official recommendations)
#
# guidance_scale: 3.5 is official default for I2V (range 3.0-5.0)
#   Values >5.0 over-saturate and increase artifacts
# num_inference_steps: 40 is official; 30 minimum for quality
# ============================================================
DEFAULT_NUM_INFERENCE_STEPS = int(os.environ.get("DEFAULT_NUM_INFERENCE_STEPS", "40"))
DEFAULT_GUIDANCE_SCALE = float(os.environ.get("DEFAULT_GUIDANCE_SCALE", "3.5"))
DEFAULT_FPS = int(os.environ.get("DEFAULT_FPS", "16"))

# ============================================================
# Post-Processing Pipeline Configuration
# ============================================================
ENABLE_POST_PROCESSING = os.environ.get("ENABLE_POST_PROCESSING", "true").lower() == "true"

# Stage 1: Face Restoration (CodeFormer primary, GFPGAN fallback)
# w=0.6 is sweet spot for AI video: good restoration, preserves identity
ENABLE_FACE_RESTORE = os.environ.get("ENABLE_FACE_RESTORE", "true").lower() == "true"
FACE_FIDELITY = float(os.environ.get("FACE_FIDELITY", "0.6"))

# Stage 2: Frame Interpolation (RIFE / FFmpeg fallback)
ENABLE_INTERPOLATION = os.environ.get("ENABLE_INTERPOLATION", "true").lower() == "true"
TARGET_FPS = int(os.environ.get("TARGET_FPS", "24"))

# Stage 3: Video Upscaling (Real-ESRGAN x4plus)
ENABLE_UPSCALE = os.environ.get("ENABLE_UPSCALE", "true").lower() == "true"
UPSCALE_FACTOR = float(os.environ.get("UPSCALE_FACTOR", "2.0"))

# Stage 4: FFmpeg Finishing (deband, denoise, sharpen, grain)
ENABLE_FFMPEG_ENHANCE = os.environ.get("ENABLE_FFMPEG_ENHANCE", "true").lower() == "true"

# ============================================================
# Timeout Configuration
# ============================================================
# Maximum time for a single video generation (seconds)
MAX_GENERATION_TIME = int(os.environ.get("MAX_GENERATION_TIME", "900"))
# Maximum time for entire job including post-processing (seconds)
MAX_JOB_TIME = int(os.environ.get("MAX_JOB_TIME", "1800"))

# ============================================================
# RunPod settings
# ============================================================
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")

# HuggingFace token (for downloading gated models)
HF_TOKEN = os.environ.get("HF_TOKEN")

# ============================================================
# Digital Ocean Spaces settings
# ============================================================
DIGITAL_OCEAN_ENDPOINT_URL = os.environ.get("DIGITAL_OCEAN_ENDPOINT_URL")
DIGITAL_OCEAN_BUCKET_ACCESS_KEY = os.environ.get("DIGITAL_OCEAN_BUCKET_ACCESS_KEY")
DIGITAL_OCEAN_BUCKET_SECRET_KEY = os.environ.get("DIGITAL_OCEAN_BUCKET_SECRET_KEY")
DIGITAL_OCEAN_BUCKET_NAME = os.environ.get("DIGITAL_OCEAN_BUCKET_NAME")
DIGITAL_OCEAN_BUCKET_URL = os.environ.get("DIGITAL_OCEAN_BUCKET_URL")
