# Whisper ASR Gradio App

A speech recognition web interface using OpenAI's Whisper model with the faster-whisper backend, FastAPI, and Gradio UI.

**Architecture:** FastAPI backend (port 10001) + Gradio frontend (port 10002)  
**Works on both CPU and GPU automatically.**

---

## What It Does

- Transcribes audio/video files to text using the Whisper model
- Supports batch transcription of multiple files simultaneously
- Extracts and transcribes files from ZIP, TAR, and TAR.GZ archives
- Supports 16+ languages: English, Finnish, Swedish, German, French, Spanish, Chinese, Japanese, Korean, Russian, Portuguese, Italian, Polish, Dutch, Arabic, and auto-detection
- Includes word-level timestamps (optional)
- Works on CPU or GPU automatically

---

## Architecture

```text
┌─────────────┐      ┌─────────────┐
│   Gradio    │ ───► │   FastAPI   │
│ (Frontend)  │ API  │ (Backend)   │
│ port 10002  │      │ port 10001  │
└─────────────┘      └─────────────┘
```

The FastAPI backend handles model loading and transcription. The Gradio frontend provides the web UI and communicates with the API.

---

## Quick Start

### Docker Compose

```bash
# Build GPU services
docker compose --profile gpu build

# Build CPU services
docker compose --profile cpu build

# Start services (GPU mode)
docker compose --profile gpu up -d

# Start services (CPU only mode)
docker compose --profile cpu up -d

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

## Usage

### Single File

Upload one audio or video file for transcription.

### Multiple Files

Upload multiple audio/video files at once. Files are processed in parallel for faster results.

### Archive

Upload a ZIP, TAR, or TAR.GZ archive containing multiple audio/video files. The archive will be extracted and all valid files will be transcribed.

---

## Requirements

- Python 3.10+
- ffmpeg
- For GPU:
  - NVIDIA GPU
  - CUDA 12.x
  - NVIDIA Container Toolkit

---

## Environment Variables

| Variable | Default | Description |
|-----------|---------|-------------|
| `WHISPER_MODEL` | `medium` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3` |
| `API_URL` | `http://localhost:10001` | FastAPI backend URL (Gradio only) |
| `API_DOCS_URL` | `http://localhost:10001/docs` | API documentation URL (Gradio only) |
| `PUBLIC_HOST` | `localhost` | Public host IP/domain for API links |

Example:

```bash
WHISPER_MODEL=tiny python api.py
```

---

## Model Sizes

| Model | Parameters | VRAM | Relative Speed |
|---------|-----------|--------|----------------|
| tiny | 39M | ~1GB | 32x |
| base | 74M | ~1GB | 16x |
| small | 244M | ~2GB | 6x |
| medium | 769M | ~5GB | 2x |
| large-v3 | 1550M | ~10GB | 1x |
| distil-large-v3 | 810M | ~5GB | ~1.5x |

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
  "model": "medium",
  "device": "cuda",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Transcribed text here"
    }
  ]
}
```

### Batch / Archive Response Format

```json
{
  "total": 3,
  "successful": 2,
  "failed": 1,
  "model": "medium",
  "device": "cuda",
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
- M4A
- AAC

### Video

- MP4
- MKV
- AVI
- MOV
- WEBM

### Archives

- ZIP
- TAR
- TAR.GZ
- TGZ

---

## Files

- `api.py` - FastAPI backend service
- `app.py` - Gradio frontend service
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker image definition
- `docker-compose.yml` - Multi-container orchestration
- `README.md` - This file

---

## Troubleshooting

### CUDA Not Available Warning

This is normal if running on CPU. The application will automatically fall back to CPU mode.

### GPU Not Detected in Docker Compose

Make sure you have the NVIDIA Container Toolkit installed:

```bash
docker compose --profile gpu up -d
```

### Slow Transcription on CPU

Use a smaller model for faster results:

```bash
WHISPER_MODEL=base python api.py
```

### Cannot Connect