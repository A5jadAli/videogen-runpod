FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

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

# Install performance optimizations (non-fatal if they fail)
# SageAttention: install from pip (v1.x is stable with Wan 2.2 BF16)
# NOTE: SageAttention 2.2.0 has known SM90 bugs on H100 (#320)
# If you need SA2, pin to a known-good commit instead of pip
RUN pip3 install --no-cache-dir sageattention || echo "SageAttention install failed, will use default attention"
RUN pip3 install --no-cache-dir cache-dit || echo "CacheDiT install failed, will run without cache acceleration"

# Patch basicsr for torchvision 0.20+ compatibility and install CodeFormer arch
RUN pip3 install --no-cache-dir lpips && \
    BASICSR_ROOT=$(python3 -c "import importlib.util; spec = importlib.util.find_spec('basicsr'); print(spec.submodule_search_locations[0])") && \
    echo "basicsr root: ${BASICSR_ROOT}" && \
    DEGRADATIONS="${BASICSR_ROOT}/data/degradations.py" && \
    if [ -f "$DEGRADATIONS" ]; then \
        sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$DEGRADATIONS" && \
        echo "Patched basicsr degradations.py for torchvision compat."; \
    fi && \
    BASICSR_ARCHS="${BASICSR_ROOT}/archs" && \
    echo "Installing CodeFormer arch files into ${BASICSR_ARCHS}/" && \
    wget -q -O "${BASICSR_ARCHS}/codeformer_arch.py" \
    https://raw.githubusercontent.com/sczhou/CodeFormer/master/basicsr/archs/codeformer_arch.py && \
    wget -q -O "${BASICSR_ARCHS}/vqgan_arch.py" \
    https://raw.githubusercontent.com/sczhou/CodeFormer/master/basicsr/archs/vqgan_arch.py && \
    python3 -c "from basicsr.archs.codeformer_arch import CodeFormer; print('CodeFormer arch import verified.')" && \
    echo "CodeFormer architecture files installed and verified."

ENV HF_HOME=/app/huggingface
ENV MODEL_CACHE_DIR=/app/models

# ============================================================
# CUDA Memory Configuration
# expandable_segments prevents fragmentation OOM on long runs
# garbage_collection_threshold proactively reclaims at 80% usage
# ============================================================
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

RUN mkdir -p /app/models /app/huggingface

# ============================================================
# Download Post-Processing Models (~2GB total — always baked in)
# ============================================================

# CodeFormer — Primary face restoration
RUN mkdir -p /app/models/codeformer && \
    wget -q -O /app/models/codeformer/codeformer.pth \
    https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth && \
    echo "CodeFormer model downloaded."

# GFPGAN v1.4 — Fallback face restoration
RUN mkdir -p /app/models/gfpgan && \
    wget -q -O /app/models/gfpgan/GFPGANv1.4.pth \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth && \
    echo "GFPGAN v1.4 model downloaded."

# Real-ESRGAN x4plus — Video upscaling
RUN mkdir -p /app/models/realesrgan && \
    wget -q -O /app/models/realesrgan/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
    echo "Real-ESRGAN x4plus model downloaded."

# Face detection models
RUN mkdir -p /app/models/facelib && \
    (wget -q -O /app/models/facelib/detection_Resnet50_Final.pth \
    https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    wget -q -O /app/models/facelib/parsing_parsenet.pth \
    https://github.com/xinntao/facexlib/releases/download/v0.1.0/parsing_parsenet.pth && \
    echo "Face detection/parsing models downloaded.") || \
    echo "WARNING: facelib model download failed. Models will auto-download at runtime."

ENV GFPGAN_MODEL_PATH=/app/models/gfpgan/GFPGANv1.4.pth
ENV FACELIB_MODELDIR=/app/models/facelib

# ============================================================
# RIFE Frame Interpolation
# ============================================================
RUN git clone https://github.com/hzwer/Practical-RIFE.git /app/rife && \
    cd /app/rife && \
    git checkout v4.22 2>/dev/null || git checkout main && \
    echo "Practical-RIFE cloned."

RUN mkdir -p /app/rife/train_log && \
    python3 -c "\
import os, sys; \
sys.path.insert(0, '/app/rife'); \
print('Attempting to download RIFE model...'); \
try: \
    from model.RIFE import Model; \
    m = Model(); \
    m.load_model('/app/rife/train_log', -1); \
    print('RIFE model loaded successfully.'); \
except Exception as e: \
    print(f'RIFE model download/load failed: {e}'); \
    print('Pipeline will use FFmpeg minterpolate fallback.'); \
" || echo "RIFE setup failed (non-fatal). FFmpeg fallback will be used."

# ============================================================
# Copy Application Code
# ============================================================
COPY main.py /app/main.py
COPY server/ /app/server/

# Create directories
RUN mkdir -p /app/temp /app/control_inputs

# ============================================================
# Environment Defaults — Research-backed for Wan 2.2 A14B
# ============================================================

# Wan 2.2 generation settings
ENV ENABLE_VAE_TILING=false
ENV ENABLE_VAE_SLICING=false
ENV ENABLE_CPU_OFFLOAD=true
ENV ENABLE_SEQUENTIAL_CPU_OFFLOAD=false
ENV ENABLE_SAGE_ATTENTION=true
ENV ENABLE_TEACACHE=true
ENV ENABLE_TORCH_COMPILE=false
ENV WAN_VERSION=2.2

# Post-processing pipeline (all enabled by default)
ENV ENABLE_POST_PROCESSING=true
ENV ENABLE_FACE_RESTORE=true
ENV FACE_FIDELITY=0.6
ENV ENABLE_INTERPOLATION=true
ENV ENABLE_UPSCALE=true
ENV ENABLE_FFMPEG_ENHANCE=true
ENV TARGET_FPS=24
ENV UPSCALE_FACTOR=2.0
ENV POST_PROCESSING_MODEL_DIR=/app/models

# Timeouts
ENV MAX_GENERATION_TIME=900
ENV MAX_JOB_TIME=1800

# Start handler
CMD ["python3", "-u", "main.py"]
