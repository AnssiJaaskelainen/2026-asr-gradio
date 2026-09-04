# syntax=docker/dockerfile:1.4  
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Create directories
RUN mkdir -p /app/models /app/caches /app/config

# Install system dependencies
# Removed redundant python/pip installs as they are in the base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY requirements.txt .
COPY api.py app.py ./
COPY config/ ./config/

# Install dependencies
RUN pip install --break-system-packages -r requirements.txt

EXPOSE 10001 10002

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:10001/health || exit 1

CMD ["python3", "/app/api.py"]