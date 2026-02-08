import os

# Base directory of project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Temp directory for generated videos
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# Model settings
T2V_MODEL_ID = os.environ.get("T2V_MODEL_ID", "Wan-AI/Wan2.1-T2V-14B-Diffusers")
I2V_MODEL_ID = os.environ.get("I2V_MODEL_ID", "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers")
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/workspace/models")

# Memory optimization flags
ENABLE_CPU_OFFLOAD = os.environ.get("ENABLE_CPU_OFFLOAD", "false").lower() == "true"
ENABLE_SEQUENTIAL_CPU_OFFLOAD = os.environ.get("ENABLE_SEQUENTIAL_CPU_OFFLOAD", "false").lower() == "true"
ENABLE_VAE_TILING = os.environ.get("ENABLE_VAE_TILING", "true").lower() == "true"

# RunPod settings
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")

# HuggingFace token (for downloading models)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Digital Ocean Spaces settings
DIGITAL_OCEAN_ENDPOINT_URL = os.environ.get("DIGITAL_OCEAN_ENDPOINT_URL")
DIGITAL_OCEAN_BUCKET_ACCESS_KEY = os.environ.get("DIGITAL_OCEAN_BUCKET_ACCESS_KEY")
DIGITAL_OCEAN_BUCKET_SECRET_KEY = os.environ.get("DIGITAL_OCEAN_BUCKET_SECRET_KEY")
DIGITAL_OCEAN_BUCKET_NAME = os.environ.get("DIGITAL_OCEAN_BUCKET_NAME")
DIGITAL_OCEAN_BUCKET_URL = os.environ.get("DIGITAL_OCEAN_BUCKET_URL")
