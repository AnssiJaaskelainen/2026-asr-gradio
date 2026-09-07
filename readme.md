# 🎙️ Whisper ASR - Audio & Video Transcription

Turn your audio and video files into text automatically. This application uses OpenAI's Whisper model and runs on your own hardware (CPU or GPU).

> 📖 **For complete documentation, advanced features, and API reference, see [full-guide.md](./full-guide.md)**

---

## 🚀 Quick Start (Windows + Docker Desktop)

### Prerequisites

| Requirement | Description |
|-------------|-------------|
| Windows | 10 or 11 |
| Docker Desktop | 4.0+ (with WSL 2 enabled) |

### Step 0: Download the Repository

Choose one of the following non-command-line methods to download the project or use cli of familiar with it:

#### Option A: Download as a ZIP File (Easiest)
1. Open the repository page on GitHub in your web browser.
2. Click the green **`<> Code`** button near the top right.
3. Select **Download ZIP**.
4. Extract the `.zip` file to your preferred location (e.g., `C:\Projects\Whisper-ASR`).

#### Option B: Clone using GitHub Desktop
1. Open **GitHub Desktop**.
2. Click **File** → **Clone repository...**
3. Select the repository from your list or paste the repository URL.
4. Choose your local folder destination and click **Clone**.

---

### Step 1: Create the transcripts folder

Create an empty folder named `transcripts` in your project directory. This is where your transcriptions will be saved.

```
📁 your-project-folder/
   ├── api.py
   ├── app.py
   ├── 📁 config/
   ├── dockerfile
   ├── docker-compose.yaml
   ├── +other needed files..
   └── 📁 transcripts/    ← Create this folder
```

### Step 2: Open Docker Desktop's terminal

1. Open **Docker Desktop** from the Start menu
2. Wait until it shows **"Docker Desktop is running"** in the system tray
3. Click the >_ Terminal in lower right hand side of the docker desktop to open inbuild terminal (Note! you might need to enable it first)

### Step 3: Start the application

```bash
cd C:\path\to\your\project-folder
docker compose up -d --build
```

First build takes some time (estimated 2-3 minutes)
* You can follow the progress on Builds tab / Active builds
* There's two active builds on going, one for API and one for the UI part
* After build Windows firewall might say something about docker network access, this needs to be allowed

### Step 4: Open the app

Open your browser and go to:

```
http://localhost:10002
```

You should see the Whisper ASR interface with a green **"API Connected"** status.

### Step 5: Transcribe your first file

1. Click **📝 Single Transcribe**
2. Upload an audio or video file
3. Click **⚡ Transcribe**
4. Your transcription appears below, and a `.txt` file is saved to the `transcripts` folder

### Stopping the application

```bash
docker compose down
```

---

## 🌟 What Can It Do?

### 📄 Single File
Upload one audio or video file and get a transcription instantly.

### 📚 Batch Processing
Upload multiple files at once. They're processed automatically in a queue.

### 📦 Archives
Upload a `.zip` or `.tar` file containing multiple media files. The app extracts and transcribes everything inside.

### 🌍 Languages
- **Auto Detect** - Automatically identify the spoken language
- **Manual Selection** - Choose from 100+ supported languages

### ⏱️ Timestamps
Enable timestamps to see when each segment was spoken:

```
[00:00.00 -> 00:05.23] Hello, welcome to the meeting.
[00:05.25 -> 00:10.45] Today we'll discuss the results.
```

### ⚡ Live Updates
Watch the transcription appear in real-time as the AI processes your file.

---

## 🤖 Available Models

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| **tiny** | 🚀🚀🚀🚀🚀 | 📉 | ~1 GB | Quick previews |
| **base** | 🚀🚀🚀🚀 | 📉📉 | ~1 GB | Fast transcription |
| **small** | 🚀🚀🚀 | 📉📉📉 | ~2 GB | General use |
| **medium** | 🚀🚀 | 📉📉📉📉 | ~3 GB | Better accuracy |
| **large-v3-turbo** | 🚀🚀🚀 | 📉📉📉📉 | ~3 GB | ⭐ Recommended |
| **large-v3** | 🚀 | 📉📉📉📉📉 | ~6 GB | Maximum accuracy |

> **Default:** `large-v3-turbo` provides the best balance of speed and accuracy.

---

## 📂 Supported Formats

### Audio
```
.mp3   .wav   .m4a   .flac   .ogg   .opus
```

### Video
```
.mp4   .avi   .mkv   .mov   .webm   .flv
```

### Archives
```
.zip   .tar   .gz   .tgz
```

---

## 🔧 Troubleshooting

### "Docker Desktop is not running"
Start Docker Desktop from the Start menu and wait for it to fully initialize.

### "Port 10001/10002 is already in use"
Stop the conflicting application or check if the app is already running:
```bash
docker compose ps
```

### "Permission denied on transcripts folder"
1. Ensure the `transcripts` folder exists before starting Docker
2. Try running Docker Desktop as Administrator

### "It's too slow!"
1. **Enable GPU:** Install NVIDIA Container Toolkit and ensure GPU is available
2. **Use a faster model:** Select `small` or `base` in the dropdown
3. **Limit CPU:** Edit `.env` and set `WHISPER_THREADS=4`

### "API shows as disconnected"
Check if containers are running:
```bash
docker compose ps
```
View logs:
```bash
docker compose logs api
```

---

## 📊 Check Application Status

View real-time logs:
```bash
docker compose logs -f
```

Restart the application:
```bash
docker compose restart
```

---

## 📖 More Information

For complete documentation including:
- REST API reference
- Python client examples
- Advanced configuration
- Production deployment guide

**See [full-guide.md](./full-guide.md)**

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Gradio](https://gradio.app/)
- [FastAPI](https://fastapi.tiangolo.com/)
