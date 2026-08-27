# Simple ASR Gradio Dockerfile - CPU & GPU
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install NVIDIA libraries for faster-whisper GPU support
RUN pip install --no-cache-dir \
    nvidia-cublas-cu12 nvidia-cudnn-cu12

# Copy and install app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]