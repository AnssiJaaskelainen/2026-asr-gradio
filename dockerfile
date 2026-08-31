# syntax=docker/dockerfile:1.4

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    python3-full \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install PyTorch (CPU by default, override for GPU in docker-compose)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_PACKAGES=torch==2.13.0 torchvision torchaudio
RUN pip install --no-cache-dir --break-system-packages ${TORCH_PACKAGES} --index-url ${TORCH_INDEX}

# Install faster-whisper and other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application files
COPY api.py app.py ./

EXPOSE 10001 10002

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:10001/health || exit 1

CMD ["python3", "/app/api.py"]