import os

# ============================================================
# CUDA Memory Configuration (set BEFORE any torch imports)
# ============================================================
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)

# Base directory of project (on Modal this is /app)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Temp directory for generated videos and control inputs
TEMP_DIR = os.path.join(BASE_DIR, "temp")
CONTROL_DIR = os.path.join(BASE_DIR, "control_inputs")

# ============================================================
# Model settings — Wan 2.2 A14B MoE Architecture
# ============================================================
WAN_VERSION = os.environ.get("WAN_VERSION", "2.2")
T2V_MODEL_ID = os.environ.get("T2V_MODEL_ID", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
I2V_MODEL_ID = os.environ.get("I2V_MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")

# ============================================================
# Model Cache Directory — points to the persistent Modal volume.
# The Wan 2.2 model (~126GB) is downloaded once into the volume.
# ============================================================
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/models")

# Post-processing models are baked into the image at /app/models
POST_PROCESSING_MODEL_DIR = os.environ.get("POST_PROCESSING_MODEL_DIR", "/app/models")

# ============================================================
# Memory + Performance optimizations
#
# On H200 (141GB) the full A14B model fits in VRAM — CPU offload is
# DISABLED by default for maximum speed. If you switch to an 80GB GPU
# (H100/A100-80GB), set ENABLE_CPU_OFFLOAD=true.
# ============================================================
ENABLE_CPU_OFFLOAD = os.environ.get("ENABLE_CPU_OFFLOAD", "false").lower() == "true"
ENABLE_SEQUENTIAL_CPU_OFFLOAD = os.environ.get("ENABLE_SEQUENTIAL_CPU_OFFLOAD", "false").lower() == "true"

# VAE tiling is BROKEN for AutoencoderKLWan (diffusers #12529). Keep disabled.
ENABLE_VAE_TILING = os.environ.get("ENABLE_VAE_TILING", "false").lower() == "true"
ENABLE_VAE_SLICING = os.environ.get("ENABLE_VAE_SLICING", "false").lower() == "true"

# SageAttention — ~2x faster attention via locked context manager.
ENABLE_SAGE_ATTENTION = os.environ.get("ENABLE_SAGE_ATTENTION", "true").lower() == "true"

# CacheDiT — TaylorSeer + DBCache ~2-3x inference speedup (MoE-aware).
ENABLE_TEACACHE = os.environ.get("ENABLE_TEACACHE", "true").lower() == "true"

# torch.compile — INCOMPATIBLE with Wan 2.2 fullgraph (#12728). Keep disabled.
ENABLE_TORCH_COMPILE = os.environ.get("ENABLE_TORCH_COMPILE", "false").lower() == "true"

# ============================================================
# Quality defaults for Wan 2.2 I2V (official recommendations)
# ============================================================
DEFAULT_NUM_INFERENCE_STEPS = int(os.environ.get("DEFAULT_NUM_INFERENCE_STEPS", "40"))
DEFAULT_GUIDANCE_SCALE = float(os.environ.get("DEFAULT_GUIDANCE_SCALE", "3.5"))
DEFAULT_FPS = int(os.environ.get("DEFAULT_FPS", "16"))

# ============================================================
# Post-Processing Pipeline Configuration
# ============================================================
ENABLE_POST_PROCESSING = os.environ.get("ENABLE_POST_PROCESSING", "true").lower() == "true"

ENABLE_FACE_RESTORE = os.environ.get("ENABLE_FACE_RESTORE", "true").lower() == "true"
FACE_FIDELITY = float(os.environ.get("FACE_FIDELITY", "0.6"))

ENABLE_INTERPOLATION = os.environ.get("ENABLE_INTERPOLATION", "true").lower() == "true"
TARGET_FPS = int(os.environ.get("TARGET_FPS", "24"))

ENABLE_UPSCALE = os.environ.get("ENABLE_UPSCALE", "true").lower() == "true"
UPSCALE_FACTOR = float(os.environ.get("UPSCALE_FACTOR", "2.0"))

ENABLE_FFMPEG_ENHANCE = os.environ.get("ENABLE_FFMPEG_ENHANCE", "true").lower() == "true"

# ============================================================
# Timeout Configuration
# ============================================================
MAX_GENERATION_TIME = int(os.environ.get("MAX_GENERATION_TIME", "1200"))   # 20 min
MAX_JOB_TIME = int(os.environ.get("MAX_JOB_TIME", "2400"))                  # 40 min

# HuggingFace token (for gated models, if needed)
HF_TOKEN = os.environ.get("HF_TOKEN")

# ============================================================
# Digital Ocean Spaces settings
# ============================================================
DIGITAL_OCEAN_ENDPOINT_URL = os.environ.get("DIGITAL_OCEAN_ENDPOINT_URL")
DIGITAL_OCEAN_BUCKET_ACCESS_KEY = os.environ.get("DIGITAL_OCEAN_BUCKET_ACCESS_KEY")
DIGITAL_OCEAN_BUCKET_SECRET_KEY = os.environ.get("DIGITAL_OCEAN_BUCKET_SECRET_KEY")
DIGITAL_OCEAN_BUCKET_NAME = os.environ.get("DIGITAL_OCEAN_BUCKET_NAME")
DIGITAL_OCEAN_BUCKET_URL = os.environ.get("DIGITAL_OCEAN_BUCKET_URL")
