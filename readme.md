# Whisper ASR Gradio App

A simple, standalone ASR (Automatic Speech Recognition) web interface using OpenAI's Whisper model with the faster-whisper backend and Gradio UI.

**Works on both CPU and GPU automatically.**

---

## What It Does

- Transcribes audio/video files to text using the Whisper model
- Supports 16+ languages: English, Finnish, Swedish, German, French, Spanish, Chinese, Japanese, Korean, Russian, Portuguese, Italian, Polish, Dutch, Arabic, and auto-detection
- Includes word-level timestamps (optional)
- Works on CPU or GPU automatically

---

## Quick Start

### Option 1: Local Installation

```bash
mkdir -p simple_asr && cd simple_asr
pip install -r requirements.txt
python app.py
```

Open http://localhost:10002 in your browser.

### Option 2: Docker

```bash
docker build -t whisper-asr .
docker run --gpus all -p 10002:10002 whisper-asr
```

Open http://localhost:10002 in your browser.

---

## Requirements

- Python 3.10+
- ffmpeg
- For GPU: NVIDIA GPU + CUDA 12.x + NVIDIA Container Toolkit

---

## Environment Variables

| Variable | Default | Description |
|-----------|---------|-------------|
| `WHISPER_MODEL` | `medium` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3` |

Example:

```bash
WHISPER_MODEL=tiny python app.py
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

## Usage

1. Open the web interface at http://localhost:10002
2. Upload an audio or video file (MP3, WAV, MP4, MKV, etc.)
3. Select a language or leave as **Auto** for automatic detection
4. Check **Include Timestamps** for word-level timing
5. Click **Transcribe** and wait for results

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

---

## Files

- `app.py` - Main Gradio application
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker image definition
- `README.md` - This file

---

## Troubleshooting

### CUDA Not Available Warning

This is normal if running on CPU. The application will automatically fall back to CPU mode.

### GPU Not Detected in Docker

Make sure you have the NVIDIA Container Toolkit installed and are using the GPU flag:

```bash
docker run --gpus all -p 10002:10002 whisper-asr
```

### Slow Transcription on CPU

Use a smaller model for faster results:

```bash
WHISPER_MODEL=base python app.py
```

---

## License

MIT License — Use freely for personal and commercial projects.