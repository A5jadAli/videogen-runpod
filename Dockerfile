# ============================================================
# VideoGen-RunPod — Wan 2.2 A14B MoE (Image-to-Video)
#
# Upgrades over Wan 2.1:
#   - MoE architecture (high-noise + low-noise experts)
#   - 65% more training images, 83% more training videos
#   - Cinematic aesthetic control (lighting, color, composition)
#   - Complex motion generation (gestures, athletics, expressions)
#   - Precise semantic compliance (multi-object, spatial accuracy)
#   - SageAttention for 40-50% faster inference
#
# GPU Requirements:
#   - Minimum: A100 80GB or H100 80GB (bf16 full quality)
#   - Recommended: H100 80GB (fastest inference)
#   - Budget: L40S 48GB or A6000 48GB (with CPU offload)
#
# No safety filters or content restrictions applied.
# ============================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies including ffmpeg, opencv deps
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

ENTRYPOINT []

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r /app/requirements.txt

# Try installing SageAttention (may fail on some CUDA versions, non-fatal)
RUN pip3 install --no-cache-dir sageattention || echo "SageAttention install failed, will use default attention"

# ============================================================
# Models are downloaded at RUNTIME on RunPod (not during build).
# Mount a RunPod network volume at /workspace so models persist
# across restarts and only download once.
#
# Wan 2.2 I2V A14B models (~30GB):
#   - Wan-AI/Wan2.2-I2V-A14B (MoE I2V model)
#
# Wan 2.2 T2V A14B models (~30GB):
#   - Wan-AI/Wan2.2-T2V-A14B (MoE T2V model)
# ============================================================

# Copy application code
COPY main.py /app/main.py
COPY server/ /app/server/

# Create directories
RUN mkdir -p /app/temp /app/control_inputs

# Environment defaults
ENV MODEL_CACHE_DIR=/workspace/models
ENV ENABLE_VAE_TILING=true
ENV ENABLE_CPU_OFFLOAD=false
ENV ENABLE_SEQUENTIAL_CPU_OFFLOAD=false
ENV ENABLE_SAGE_ATTENTION=true
ENV ENABLE_TORCH_COMPILE=false
ENV WAN_VERSION=2.2

# Start handler
CMD ["python3", "-u", "main.py"]
