# ============================================================
# VideoGen-RunPod Dockerfile
# Wan 2.1 14B Text-to-Video + Image-to-Video
# No safety filters or content restrictions
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

ENTRYPOINT []

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# ============================================================
# Models are downloaded at RUNTIME on RunPod (not during build).
# Mount a RunPod network volume at /workspace so models persist
# across restarts and only download once (~56GB total).
# ============================================================

# Copy application code
COPY main.py /app/main.py
COPY server/ /app/server/

# Create temp directory
RUN mkdir -p /app/temp

# Environment defaults
ENV MODEL_CACHE_DIR=/workspace/models
ENV ENABLE_VAE_TILING=true
ENV ENABLE_CPU_OFFLOAD=false
ENV ENABLE_SEQUENTIAL_CPU_OFFLOAD=false

# Start handler - model loads in main.py at import time
CMD ["python3", "-u", "main.py"]
