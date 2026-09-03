# Whisper ASR Gradio App

A speech recognition web interface using OpenAI's Whisper model with a faster-whisper backend (CTranslate2 optimized), FastAPI, and a Gradio UI.

**Architecture:** FastAPI backend (port 10001) + Gradio frontend (port 10002)  
**Optimized for CPU with OpenVINO acceleration on Intel processors.**

---

## What It Does

- Transcribes audio and video files to text using Whisper
- Supports batch transcription of multiple files simultaneously
- Extracts and transcribes files from ZIP, TAR, TAR.GZ, and TGZ archives
- Supports 16+ languages including English, Finnish, Swedish, German, French, Spanish, Chinese, Japanese, Korean, Russian, Portuguese, Italian, Polish, Dutch, Arabic, and auto-detection
- Optional segment-level timestamps
- Automatic CPU/GPU hardware detection
- Full multi-threaded CPU inference using faster-whisper (CTranslate2)
- OpenVINO acceleration for Intel CPUs (approximately 20–40% faster)

---

## Architecture

```text
┌─────────────┐      ┌─────────────┐
│   Gradio    │ ───► │   FastAPI   │
│ (Frontend)  │ API  │ (Backend)   │
│ port 10002  │      │ port 10001  │
└─────────────┘      └─────────────┘
```

The FastAPI backend handles model loading and transcription using faster-whisper (CTranslate2 optimized). The Gradio frontend provides the web UI and communicates with the API.

---

## Quick Start

### Docker Compose

```bash
# Build and start services
docker compose up -d --build

# Stop services
docker compose down

# Stop and remove volumes
docker compose down -v
```

Available endpoints:

- Gradio UI: http://localhost:10002
- API Docs: http://localhost:10001/docs

### Local Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run both services:

```bash
# Terminal 1 - FastAPI backend
python api.py

# Terminal 2 - Gradio frontend
python app.py
```

Open `http://localhost:10002` in your browser.

---

## Performance

The application automatically detects available hardware and configures optimal settings.

### faster-whisper (CTranslate2) Benefits

- Full multi-threaded CPU inference using all available CPU cores
- OpenVINO acceleration on Intel CPUs
- Automatic GPU acceleration when CUDA is available
- Int8 quantization for faster CPU inference

### Hardware Detection

On startup the application detects:

- Number of CPU cores
- GPU availability
- Intel CPU support for OpenVINO
- Memory constraints for batch sizing

Example CPU output:

```text
Hardware detected: CPU | Cores: 64 | Threads: 64 | Workers: 4
OpenVINO: ENABLED | Device: CPU
Backend: faster-whisper + OpenVINO
```

Example GPU output:

```text
Hardware detected: CUDA | Cores: 64 | Threads: 64 | Workers: 4
Backend: faster-whisper
```

---

## Usage

### Single File

Upload one audio or video file for transcription.

### Multiple Files

Upload multiple audio or video files at once. Files are processed in parallel.

### Archive

Upload a ZIP, TAR, TAR.GZ, or TGZ archive containing media files.

---

## Requirements

### Required

- Python 3.10+
- FFmpeg

### Optional (GPU)

- NVIDIA GPU
- CUDA 12.x
- NVIDIA Container Toolkit

---

## Environment Variables

| Variable | Default | Description |
|----------|----------|-------------|
| `WHISPER_MODEL` | `large-v3-turbo` | Model size (`tiny`, `base`, `small`, `medium`, `large-v3-turbo`, `large-v3`) |
| `WHISPER_THREADS` | `0` | Number of CPU threads (auto when 0) |
| `USE_OPENVINO` | `true` | Enable OpenVINO optimization |
| `OPENVINO_DEVICE` | `CPU` | OpenVINO device (`CPU`, `GPU`, `NPU`) |
| `API_URL` | `http://localhost:10001` | Backend URL |
| `API_DOCS_URL` | `http://localhost:10001/docs` | API documentation URL |
| `PUBLIC_HOST` | `localhost` | Public API hostname |
| `MAX_FILE_SIZE_MB` | `500` | Maximum file size |
| `MAX_BATCH_SIZE` | `20` | Maximum files per batch |
| `API_MEMORY_LIMIT` | `16g` | Memory limit for API container |

### Examples

```bash
# Use smaller model
WHISPER_MODEL=small docker compose up -d

# Disable OpenVINO
USE_OPENVINO=false docker compose up -d

# Limit CPU threads
WHISPER_THREADS=16 docker compose up -d

# Use base model
WHISPER_MODEL=base docker compose up -d
```

---

## Model Sizes

| Model | Parameters | Memory | Relative Speed |
|--------|------------|---------|----------------|
| tiny | 39M | ~1 GB | Fastest |
| base | 74M | ~1 GB | Fast |
| small | 244M | ~2 GB | Moderate |
| medium | 769M | ~3 GB | Balanced |
| large-v3-turbo | 809M | ~3 GB | 4× faster than large-v3 |
| large-v3 | 1550M | ~6 GB | Most accurate |

Models are downloaded automatically from Hugging Face on first use.

**Recommendation:** `large-v3-turbo` provides an excellent balance between speed and accuracy.

---

## API Endpoints

### Health Check

```bash
curl http://localhost:10001/health
```

### Transcribe Single File

```bash
curl -X POST "http://localhost:10001/transcribe?language=auto&return_timestamps=false" \
  -F "file=@audio.mp3"
```

### Transcribe Multiple Files

```bash
curl -X POST "http://localhost:10001/transcribe-batch?language=auto&return_timestamps=false" \
  -F "files=@file1.mp3" \
  -F "files=@file2.mp3" \
  -F "files=@file3.wav"
```

### Transcribe Archive

```bash
curl -X POST "http://localhost:10001/transcribe-archive?language=auto&return_timestamps=false" \
  -F "archive=@files.zip"
```

---

## Response Format

```json
{
  "text": "Transcribed text here",
  "language": "en",
  "language_probability": 0.99,
  "model": "large-v3-turbo",
  "device": "CPU",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Transcribed text here"
    }
  ]
}
```

### Batch / Archive Response

```json
{
  "total": 3,
  "successful": 2,
  "failed": 1,
  "model": "large-v3-turbo",
  "device": "CPU",
  "results": [
    {
      "filename": "audio1.mp3",
      "text": "Transcribed text",
      "language": "en",
      "language_probability": 0.99,
      "segments": [],
      "success": true,
      "error": null
    }
  ]
}
```

---

## Supported File Formats

### Audio

- MP3
- WAV
- FLAC
- OGG
- OPUS
- M4A
- AAC

### Video

- MP4
- MKV
- AVI
- MOV
- WEBM
- FLV

### Archives

- ZIP
- TAR
- TAR.GZ
- TGZ

---

## Project Files

- `api.py` - FastAPI backend service
- `app.py` - Gradio frontend service
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker image definition
- `docker-compose.yml` - Multi-container orchestration
- `README.md` - Project documentation

---

## Troubleshooting

### Model Download Slow

Models are downloaded from Hugging Face on first use.

```yaml
volumes:
  - ./models:/app/models
```

### CUDA Not Available

The application automatically falls back to CPU mode with OpenVINO optimization where available.

### GPU Not Detected in Docker

```bash
which nvidia-container-toolkit

sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

### High CPU Usage

Expected behavior for fastest transcription.

```bash
WHISPER_THREADS=16 docker compose up -d
```

### Cannot Connect to API

```bash
curl http://localhost:10001/health
```

### Out of Memory

```bash
WHISPER_MODEL=small MAX_BATCH_SIZE=5 docker compose up -d
```

### OpenVINO Not Working

- OpenVINO is only enabled on Intel CPUs by default
- Check logs for `OpenVINO: ENABLED`
- Disable manually with `USE_OPENVINO=false`
- AMD and ARM systems disable OpenVINO automatically
