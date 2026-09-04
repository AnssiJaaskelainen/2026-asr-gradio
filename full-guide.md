# 🎙️ Whisper ASR - Audio & Video Transcription

Turn your audio and video files into text automatically. This application uses OpenAI's Whisper model, optimized to run fast on your own hardware (CPU or GPU).

---

## 🚀 Quick Start (Windows + Docker Desktop)

This guide is for users with **Windows 10/11** and **Docker Desktop** installed.

### Prerequisites

Before starting, ensure you have:

| Requirement | Version | Download |
|-------------|---------|----------|
| Windows | 10 or 11 | [microsoft.com](https://www.microsoft.com/windows) |
| Docker Desktop | 4.0+ | [docker.com](https://www.docker.com/products/docker-desktop) |
| WSL 2 | Latest | [Enable in Windows Features](https://docs.microsoft.com/en-us/windows/wsl/install) |

> **Note:** Docker Desktop requires WSL 2 to be enabled. If you haven't set it up yet, Docker Desktop will prompt you to install it on first launch.

---

### Step 1: Create the transcripts folder

Your transcriptions need a place to be saved. Create an empty folder in your project directory:

1. Open **File Explorer** and navigate to this project folder
2. Right-click in an empty area
3. Select **New** → **Folder**
4. Name the folder exactly: `transcripts`

```
📁 your-project-folder/
   ├── 📁 api.py
   ├── 📁 app.py
   ├── 📁 config/
   ├── 📁 dockerfile
   ├── 📁 docker-compose.yaml
   └── 📁 transcripts/    ← Create this folder
```

---

### Step 2: Open Docker Desktop's built-in terminal

Docker Desktop includes a built-in terminal that's pre-configured and ready to use:

1. Open **Docker Desktop** from the Start menu
2. Wait until it shows **"Docker Desktop is running"** in the system tray (bottom-right corner)
3. Click the **Docker Desktop** icon in the system tray
4. Right-click the icon and select **"Open in terminal"**

   > 💡 **Tip:** You can also press `Ctrl + Shift + ,` in Docker Desktop to open Settings, then go to **General** and enable "Open Docker Desktop at Windows startup" for convenience.

A terminal window will open with Docker context already configured.

---

### Step 3: Navigate to your project folder

In the Docker Desktop terminal, navigate to where you extracted the project files:

```bash
cd C:\path\to\your\project-folder
```

> **Note:** If your files are in a different location, adjust the path above. You can drag and drop the folder into the terminal to auto-fill the path.

---

### Step 4: Start the application

Build and start the application:

```bash
docker compose up -d --build
```

Expected output:
```
[+] Building 15.0s
[+] Running 2/2
 ✔ Network whisper-asr_default    Created
 ✔ Container whisper-asr-api      Started
 ✔ Container whisper-asr-gradio   Started
```

---

### Step 5: Open the application

1. Wait about 30 seconds for the application to initialize (first run downloads the AI model)

2. Open your web browser and go to:

   ```
   http://localhost:10002
   ```

3. You should see the Whisper ASR interface with a green **"API Connected"** status.

---

### Step 6: Transcribe your first file

1. Click **📝 Single Transcribe** tab
2. Click the **File** upload area and select an audio or video file
3. (Optional) Choose a language from the dropdown
4. Click **⚡ Transcribe**
5. Wait for the transcription to complete
6. Your result will appear below, and a `.txt` file will be saved to the `transcripts` folder

---

### Stopping the application

When you're done, stop the application to free up resources:

```bash
docker compose down
```

To remove everything including saved transcriptions:
```bash
docker compose down -v
```

---

### Alternative: Using PowerShell

If you prefer using PowerShell instead of Docker Desktop's terminal:

1. Open **PowerShell** as Administrator
2. Navigate to your project folder:
   ```powershell
   cd C:\path\to\your\project-folder
   ```
3. Ensure Docker Desktop is running (check the system tray)
4. Run the same commands as above:
   ```powershell
   docker compose up -d --build
   ```

---

## 🌟 Features

### 📄 Single File Processing

Upload a single audio or video file and receive a transcription instantly.

### 📚 Batch Processing

Upload multiple files at once. They will be processed automatically in a queue.

### 📦 Archive Support

Upload compressed archives containing multiple files:

| Format | Extension | Example |
|--------|-----------|---------|
| ZIP Archive | `.zip` | `recordings.zip` |
| TAR Archive | `.tar`, `.tar.gz`, `.tgz` | `audio.tar.gz` |

The application automatically extracts the archive and transcribes all supported media files inside.

### 🌍 Language Support

- **Auto Detect** - Automatically identify the spoken language
- **Manual Selection** - Choose from 100+ supported languages
- **Mixed Languages** - Handle files with multiple languages

### ⏱️ Timestamps

Enable timestamps to see when each segment was spoken:

```
[00:00.00 -> 00:05.23] Hello, welcome to the meeting.
[00:05.25 -> 00:10.45] Today we'll be discussing the quarterly results.
```

### ⚡ Live Updates

Watch the transcription appear in real-time as the AI processes your file.

---

## 🤖 Available Models

Choose the right model for your needs:

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| **tiny** | 🚀🚀🚀🚀🚀 | 📉 | ~1 GB | Quick previews, testing |
| **base** | 🚀🚀🚀🚀 | 📉📉 | ~1 GB | Fast transcription |
| **small** | 🚀🚀🚀 | 📉📉📉 | ~2 GB | General use |
| **medium** | 🚀🚀 | 📉📉📉📉 | ~3 GB | Better accuracy |
| **large-v3-turbo** | 🚀🚀🚀 | 📉📉📉📉 | ~3 GB | ⭐ Recommended |
| **large-v3** | 🚀 | 📉📉📉📉📉 | ~6 GB | Maximum accuracy |

> **Default:** `large-v3-turbo` provides the best balance of speed and accuracy for most use cases.

---

## 📂 Supported File Formats

### Audio Files

```
.mp3   .wav   .m4a   .flac   .ogg   .opus
```

### Video Files

```
.mp4   .avi   .mkv   .mov   .webm   .flv
```

### Archives

```
.zip   .tar   .gz   .tgz
```


---

## 🔌 REST API Reference

The application includes a full REST API for programmatic access to transcription features.

### Base URL

```
http://localhost:10001
```

> **Note:** The API runs inside the Docker container on port 10001. The Gradio UI (port 10002) connects to this API internally.

---

### Authentication

Currently, the API does not require authentication. For production deployments, consider adding authentication via:

- API keys (via reverse proxy like Nginx)
- OAuth 2.0
- Basic Auth

---

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/models` | List available models |
| `GET` | `/languages` | List supported languages |
| `POST` | `/transcribe` | Transcribe a single file |
| `POST` | `/transcribe-batch` | Transcribe multiple files |
| `POST` | `/transcribe-archive` | Transcribe files from archive |
| `GET` | `/transcripts` | List saved transcriptions |
| `GET` | `/transcripts/archive` | Download all transcriptions as ZIP |

---

### Health Check

Check if the API is running and get system information.

**Request:**
```bash
curl http://localhost:10001/health
```

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "default_model": "large-v3-turbo"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"healthy"` if running |
| `device` | string | `"cuda"` for GPU, `"cpu"` for CPU |
| `default_model` | string | Default Whisper model ID |

---

### List Available Models

**Request:**
```bash
curl http://localhost:10001/models
```

**Response:**
```json
{
  "models": [
    {
      "id": "tiny",
      "name": "Tiny (39M)",
      "description": "Whisper tiny == Fastest but not that accurate"
    },
    {
      "id": "base",
      "name": "Base (74M)",
      "description": "Whisper base == Good balance of speed"
    }
  ],
  "default": "large-v3-turbo"
}
```

---

### List Supported Languages

**Request:**
```bash
curl http://localhost:10001/languages
```

**Response:**
```json
{
  "languages": [
    {"code": "auto", "name": "Auto Detect"},
    {"code": "en", "name": "English"},
    {"code": "fi", "name": "Finnish"}
  ]
}
```

---

### Transcribe Single File

Transcribe an audio or video file.

**Request:**
```bash
curl -X POST http://localhost:10001/transcribe \
  -F "file=@audio.mp3" \
  -F "language=en" \
  -F "return_timestamps=false" \
  -F "model_id=large-v3-turbo"
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | file | Yes | - | Audio or video file |
| `language` | string | No | `auto` | Language code or `auto` |
| `return_timestamps` | boolean | No | `false` | Include timestamps in output |
| `model_id` | string | No | `large-v3-turbo` | Whisper model to use |
| `stream` | boolean | No | `false` | Enable streaming response |

**Response:**
```json
{
  "text": "Hello, this is a sample transcription of the audio file.",
  "language": "en",
  "language_probability": 0.99,
  "model": "large-v3-turbo",
  "device": "cuda",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is a sample"
    },
    {
      "start": 2.5,
      "end": 5.0,
      "text": "transcription of the audio file."
    }
  ],
  "request_id": "a1b2c3d4",
  "file_path": "/transcripts/20240115_143022/audio.txt"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Full transcription text |
| `language` | string | Detected or specified language |
| `language_probability` | float | Confidence score (0.0 - 1.0) |
| `model` | string | Model used for transcription |
| `device` | string | Device used (`cuda` or `cpu`) |
| `segments` | array | List of text segments with timestamps |
| `request_id` | string | Unique request identifier |
| `file_path` | string | Path where transcription was saved |

---

### Transcribe with Timestamps

**Request:**
```bash
curl -X POST http://localhost:10001/transcribe \
  -F "file=@meeting.mp4" \
  -F "return_timestamps=true"
```

**Response (with timestamps):**
```json
{
  "text": "Hello everyone. Welcome to the meeting. Today we will discuss the quarterly results.",
  "language": "en",
  "language_probability": 0.98,
  "model": "large-v3-turbo",
  "device": "cuda",
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "text": "Hello everyone."
    },
    {
      "start": 3.4,
      "end": 7.1,
      "text": "Welcome to the meeting."
    },
    {
      "start": 7.3,
      "end": 12.5,
      "text": "Today we will discuss the quarterly results."
    }
  ],
  "request_id": "e5f6g7h8",
  "file_path": "/transcripts/20240115_143522/meeting.txt"
}
```

---

### Streaming Transcription

For real-time transcription with live updates, use streaming mode:

**Request:**
```bash
curl -X POST http://localhost:10001/transcribe \
  -F "file=@long_recording.mp3" \
  -F "stream=true" \
  -N
```

**Response (Server-Sent Events):**
```
data: {"type": "metadata", "duration": 3600.0, "language": "en", "language_probability": 0.97}

data: {"type": "segment", "start": 0.0, "end": 3.5, "text": "Hello, welcome.", "index": 0}

data: {"type": "segment", "start": 3.5, "end": 7.2, "text": "Today we have a lot to cover.", "index": 1}

data: {"type": "segment", "start": 7.2, "end": 12.0, "text": "Let's begin with the introduction.", "index": 2}

data: {"type": "done", "text": "Hello, welcome. Today we have a lot to cover. Let's begin with the introduction.", "success": true, "error": null}
```

**Event Types:**

| Type | Description |
|------|-------------|
| `metadata` | Initial info: duration, detected language |
| `segment` | Each transcribed segment as it's completed |
| `done` | Final response with full text |

---

### Batch Transcription

Transcribe multiple files in a single request.

**Request:**
```bash
curl -X POST http://localhost:10001/transcribe-batch \
  -F "files=@file1.mp3" \
  -F "files=@file2.mp3" \
  -F "files=@file3.mp3" \
  -F "language=en"
```

**Response:**
```json
{
  "total": 3,
  "successful": 2,
  "failed": 1,
  "model": "large-v3-turbo",
  "device": "cuda",
  "results": [
    {
      "filename": "file1.mp3",
      "text": "Transcription of file 1...",
      "language": "en",
      "language_probability": 0.99,
      "duration": 120.5,
      "segments": [...],
      "success": true,
      "file_path": "/transcripts/20240115_144000/file1.txt"
    },
    {
      "filename": "file2.mp3",
      "text": "Transcription of file 2...",
      "language": "en",
      "language_probability": 0.98,
      "duration": 95.3,
      "segments": [...],
      "success": true,
      "file_path": "/transcripts/20240115_144000/file2.txt"
    },
    {
      "filename": "file3.mp3",
      "success": false,
      "error": "Audio file is corrupted"
    }
  ],
  "request_id": "i9j0k1l2"
}
```

---

### Archive Transcription

Upload a ZIP or TAR archive containing multiple audio/video files.

**Request:**
```bash
curl -X POST http://localhost:10001/transcribe-archive \
  -F "archive=@recordings.zip" \
  -F "language=auto"
```

**Supported Archive Formats:**
- `.zip`
- `.tar`
- `.tar.gz`
- `.tgz`

**Response:**
```json
{
  "total": 5,
  "successful": 4,
  "results": [
    {
      "filename": "interview_01.mp3",
      "text": "...",
      "language": "en",
      "success": true,
      "file_path": "/transcripts/20240115_144500/interview_01.txt"
    },
    {
      "filename": "interview_02.mp3",
      "text": "...",
      "language": "en",
      "success": true,
      "file_path": "/transcripts/20240115_144500/interview_02.txt"
    }
  ],
  "request_id": "m3n4o5p6"
}
```

---

### List Saved Transcriptions

Get a list of all saved transcription sessions.

**Request:**
```bash
curl http://localhost:10001/transcripts
```

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "20240115_144500",
      "files": ["interview_01.txt", "interview_02.txt"],
      "created_at": "20240115_144500"
    },
    {
      "session_id": "20240115_143022",
      "files": ["audio.txt", "meeting.txt"],
      "created_at": "20240115_143022"
    }
  ]
}
```

---

### Download All Transcriptions

Download all transcriptions as a ZIP archive.

**Request:**
```bash
curl -O http://localhost:10001/transcripts/archive
```

Or save to a specific file:
```bash
curl -o all_transcriptions.zip http://localhost:10001/transcripts/archive
```

---

## 💻 Python API Client

Use the built-in Python client for easier integration:

```python
import httpx

# Initialize client
client = httpx.Client(base_url="http://localhost:10001", timeout=300)

# Health check
health = client.get("/health").json()
print(f"Status: {health['status']}, Device: {health['device']}")

# Transcribe a file
with open("audio.mp3", "rb") as f:
    response = client.post("/transcribe", files={"file": f})
    result = response.json()
    print(result["text"])
```

### Async Client Example

```python
import asyncio
import httpx

async def transcribe_async(file_path: str):
    async with httpx.AsyncClient(base_url="http://localhost:10001", timeout=300) as client:
        with open(file_path, "rb") as f:
            response = await client.post("/transcribe", files={"file": f})
            return response.json()

# Usage
result = asyncio.run(transcribe_async("audio.mp3"))
print(result["text"])
```

### Streaming Client Example

```python
import httpx
import json

client = httpx.Client(base_url="http://localhost:10001", timeout=None)

with open("audio.mp3", "rb") as f:
    with client.stream("POST", "/transcribe", files={"file": f, "stream": "true"}) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event["type"] == "segment":
                    print(f"[{event['start']:.1f}s] {event['text']}")
                elif event["type"] == "done":
                    print(f"\nTotal: {event['text'][:100]}...")
```

---

## 🛠️ API Configuration

Configure the API via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FILE_SIZE_MB` | `500` | Maximum upload size per file (MB) |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `WHISPER_THREADS` | `0` | CPU threads (0 = auto) |
| `CPU_OPTIMIZED` | `true` | Enable CPU optimization |
| `API_MEMORY_LIMIT` | `16g` | Container memory limit |

---

## 📡 API Rate Limits

Currently, there are no rate limits enforced. For production use, consider adding rate limiting via:

- **Nginx:** Use `limit_req_module`
- **Traefik:** Use rate limiting middleware
- **API Gateway:** AWS API Gateway, Kong, etc.

---

## 🔒 Security Considerations

For production deployments:

1. **Add authentication** - Protect endpoints with API keys or OAuth
2. **Enable HTTPS** - Use a reverse proxy with TLS termination
3. **Limit file sizes** - Set `MAX_FILE_SIZE_MB` appropriately
4. **Restrict CORS** - Set `CORS_ORIGINS` to specific domains
5. **Add rate limiting** - Prevent abuse
6. **Scan uploads** - Validate file types before processing

Example Nginx configuration for production:

```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20;

    # Proxy to API container
    location / {
        proxy_pass http://localhost:10001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 🐛 API Error Responses

The API returns standard HTTP status codes:

| Code | Meaning | Example |
|------|---------|---------|
| `200` | Success | Transcription completed |
| `400` | Bad Request | Invalid file format |
| `404` | Not Found | Resource doesn't exist |
| `413` | Payload Too Large | File exceeds `MAX_FILE_SIZE_MB` |
| `500` | Server Error | Transcription failed |

**Error Response Format:**
```json
{
  "detail": "No supported filetypes found inside the uploaded archive file."
}
```


## ⚙️ Configuration

Customize the application by editing the `.env` file in your project folder:

```env
# Performance Settings
CPU_OPTIMIZED=true          # Enable CPU optimization (true/false)
WHISPER_THREADS=0           # CPU cores to use (0 = auto-detect)
MAX_FILE_SIZE_MB=500        # Maximum upload size per file

# Resource Limits
API_MEMORY_LIMIT=16g        # Maximum memory for transcription engine

# System Settings
UID=1000                    # User ID (Linux only)
GID=1000                    # Group ID (Linux only)
TZ=UTC                      # Timezone
```

### Recommended Settings

| Scenario | Settings |
|----------|----------|
| **Fast transcription** | `WHISPER_THREADS=4`, `CPU_OPTIMIZED=true` |
| **Long files** | `MAX_FILE_SIZE_MB=1000` |
| **Limited RAM (8GB)** | `API_MEMORY_LIMIT=8g`, use `small` model |
| **Maximum quality** | Use `large-v3` model, ensure GPU available |

---

## 🔧 Troubleshooting

### "Docker Desktop is not running"

**Symptom:** Error message about Docker daemon not being accessible.

**Solution:**
1. Start Docker Desktop from the Start menu
2. Wait for the whale icon in the system tray to stop animating
3. Verify with: `docker info`

---

### "Port 10001/10002 is already in use"

**Symptom:** Error message about binding to ports.

**Solution:**
1. Check what's using the ports:
   ```bash
   netstat -ano | findstr "10001"
   netstat -ano | findstr "10002"
   ```
2. Stop the conflicting application or change the ports in `docker-compose.yaml`

---

### "I get a timeout error when uploading large files"

**Symptom:** Browser shows timeout, but transcription may still be running.

**Solution:**
1. Check the `transcripts` folder - files may still be created
2. Increase timeout in your browser settings
3. For very large files, consider splitting them into smaller chunks

---

### "Permission denied on the transcripts folder"

**Symptom:** Cannot save transcription files.

**Solution:**
1. Ensure the `transcripts` folder exists before starting Docker
2. Check folder permissions in Docker Desktop terminal:
   ```bash
   ls -la transcripts/
   ```
3. Try running Docker Desktop as Administrator

---

### "It's too slow!"

**Symptom:** Transcription takes a very long time.

**Solutions (try in order):**

1. **Enable GPU acceleration:**
   - Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
   - Ensure NVIDIA drivers are up to date

2. **Use a faster model:**
   - Go to Settings → Model → Select `small` or `base`

3. **Limit CPU usage:**
   - Edit `.env`: `WHISPER_THREADS=4`

4. **Enable CPU optimization:**
   - Edit `.env`: `CPU_OPTIMIZED=true`

---

### "The container time is different from my host time"

**Symptom:** Timestamps in transcriptions don't match your local time.

**Solution:**
1. The container syncs with your system time automatically
2. If incorrect, enable automatic time in Windows:
   - Settings → Time & Language → Date & Time
   - Enable "Set time automatically"

---

### "API shows as disconnected"

**Symptom:** Gradio interface shows red "API Disconnected" status.

**Solution:**
1. Check if containers are running:
   ```bash
   docker compose ps
   ```
2. View API logs:
   ```bash
   docker compose logs api
   ```
3. Restart the services:
   ```bash
   docker compose restart
   ```

---

## 📊 Checking Application Status

View real-time logs to monitor the application:

```bash
# View all logs
docker compose logs -f

# View only API logs
docker compose logs -f api

# View only Gradio logs
docker compose logs -f gradio
```

Check which models are downloaded:

```bash
docker compose exec api ls -la /app/models/
```

---

## 🗑️ Maintenance

### Clear cached models

```bash
docker compose down
rm -rf ./models/*
docker compose up -d --build
```

### Clear all transcriptions

```bash
rm -rf ./transcripts/*
mkdir transcripts
```

### Full reset

```bash
docker compose down -v
rm -rf ./models/ ./caches/ ./transcripts/
mkdir transcripts
docker compose up -d --build
```

---

## 📄 License

This project uses OpenAI's Whisper model. Please review OpenAI's [usage policies](https://openai.com/policies/usage-policies) for commercial use.

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Optimized inference engine
- [Gradio](https://gradio.app/) - Web interface framework
- [FastAPI](https://fastapi.tiangolo.com/) - API framework