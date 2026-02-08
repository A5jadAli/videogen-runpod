import os

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
T2V_MODEL_ID = os.environ.get("T2V_MODEL_ID", "Wan-AI/Wan2.2-T2V-A14B")
I2V_MODEL_ID = os.environ.get("I2V_MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B")

# Model cache (persist across restarts on RunPod network volume)
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/workspace/models")

# ============================================================
# Memory + Performance optimizations
# ============================================================
ENABLE_CPU_OFFLOAD = os.environ.get("ENABLE_CPU_OFFLOAD", "false").lower() == "true"
ENABLE_SEQUENTIAL_CPU_OFFLOAD = os.environ.get("ENABLE_SEQUENTIAL_CPU_OFFLOAD", "false").lower() == "true"
ENABLE_VAE_TILING = os.environ.get("ENABLE_VAE_TILING", "true").lower() == "true"

# SageAttention — ~40-50% faster inference, negligible quality loss
ENABLE_SAGE_ATTENTION = os.environ.get("ENABLE_SAGE_ATTENTION", "true").lower() == "true"

# torch.compile — 10-20% faster after warmup, slow first run
ENABLE_TORCH_COMPILE = os.environ.get("ENABLE_TORCH_COMPILE", "false").lower() == "true"

# ============================================================
# Quality defaults for Wan 2.2
# ============================================================
# Wan 2.2 benefits from higher inference steps than 2.1
DEFAULT_NUM_INFERENCE_STEPS = int(os.environ.get("DEFAULT_NUM_INFERENCE_STEPS", "50"))
DEFAULT_GUIDANCE_SCALE = float(os.environ.get("DEFAULT_GUIDANCE_SCALE", "5.0"))
DEFAULT_FPS = int(os.environ.get("DEFAULT_FPS", "16"))

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
