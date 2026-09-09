# 🎙️ Whisper ASR - Audio & Video Transcription

Turn your audio and video files into text automatically. This application uses OpenAI's Whisper model and runs on your own hardware (CPU or GPU).

📖 For complete documentation, advanced features, and API reference, see **full-guide.md**

---

# 🚀 Quick Start (Windows + Docker Desktop)

## Prerequisites

| Requirement | Description |
|------------|-------------|
| Windows | 10 or 11 |
| Docker Desktop | 4.0+ (with WSL 2 enabled) |
| NVIDIA GPU (Optional) | For much faster transcription (requires NVIDIA Drivers + NVIDIA Container Toolkit) |

---

## Step 0: Download the Repository

Choose one of the following methods:

### Option A: Download as a ZIP File (Easiest)

1. Open the repository page on GitHub.
2. Click the green **<> Code** button → **Download ZIP**.
3. Extract to your preferred location.

### Option B: Clone using GitHub Desktop

1. Open GitHub Desktop.
2. Click **File** → **Clone repository...**
3. Paste the repository URL and click **Clone**.

---

## Step 1: Create the `transcripts` Folder

Create an empty folder named `transcripts` in your project directory.

```text
📁 your-project-folder/
├── api.py
├── app.py
├── 📁 config/
├── dockerfile
├── dockerfile.gpu
├── docker-compose.yaml
└── 📁 transcripts/    ← Create this folder
├── ++other related files
```

---

## Step 2: Open Docker Desktop's Terminal

1. Open Docker Desktop.
2. Click the **>_ Terminal** button in the lower right-hand corner to open the built-in terminal.

---

## Step 3: Start the Application

> **Important:** Choose **ONE** of the following options based on your hardware. Do **not** run both.

### Option A: Use NVIDIA GPU (Recommended for Speed 🚀)

Use this if you have an NVIDIA graphics card and the NVIDIA Container Toolkit installed. On Linux if you can run nvtop and see some results you should be good to go

```bash
docker compose up -d --build api-gpu gradio-gpu
# Note! You will get this if running gpu enabled version without a gpu
# Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]
```

### Option B: Use CPU (Standard 💻)

Use this if you do not have an NVIDIA GPU.

```bash
docker compose up -d --build api-cpu gradio-cpu
```

> **Note:** The first build may take several minutes. Once completed, you should see two running containers in Docker Desktop.

---

## Step 4: Open the App

Open your browser and navigate to:

```text
http://localhost:10002
```

You should see the Whisper ASR interface with a green **"API Connected"** status indicator.

---

## Step 5: Transcribe Your First File

1. Click **📝 Single Transcribe**
2. Upload an audio or video file
3. Click **⚡ Transcribe**
4. Your transcription will appear below
5. A `.txt` file will also be saved in the `transcripts` folder

---

# Stopping the Application

In the terminal, run:

```bash
docker compose down
```

---

# 🌟 What Can It Do?

## 📄 Single File

Upload one audio or video file and receive a transcription instantly.

## 📚 Batch Processing

Upload multiple files at once. Files are processed automatically in a queue.

## 📦 Archives

Upload a `.zip` or `.tar` archive containing multiple media files. The application extracts and transcribes all supported files inside.

## ⏱️ Timestamps

Enable timestamps to see exactly when each segment was spoken.

---

# 🤖 Available Models

| Model | Speed | Accuracy | Memory | Best For |
|---------|---------|---------|---------|---------|
| tiny | 🚀🚀🚀🚀🚀 | 📉 | ~1 GB | Quick previews |
| base | 🚀🚀🚀🚀 | 📉📉 | ~1 GB | Fast transcription |
| small | 🚀🚀🚀 | 📉📉📉 | ~2 GB | General use |
| medium | 🚀🚀 | 📉📉📉📉 | ~3 GB | Better accuracy |
| large-v3-turbo | 🚀🚀🚀 | 📉📉📉📉 | ~3 GB | ⭐ Recommended |
| large-v3 | 🚀 | 📉📉📉📉📉 | ~6 GB | Maximum accuracy |

---

# 🔧 Troubleshooting

## "Port 10001/10002 is already in use"

This typically occurs when both CPU and GPU versions are started simultaneously.

Stop all containers:

```bash
docker compose down
```

Then start only one version:

```bash
docker compose up -d --build api-gpu gradio-gpu
```

or

```bash
docker compose up -d --build api-cpu gradio-cpu
```

---

## "GPU Not Detected" or "Transcription Is Slow"

Verify that:

- NVIDIA drivers are installed
- NVIDIA Container Toolkit is installed
- You started the GPU version using:

```bash
docker compose up -d --build api-gpu gradio-gpu
```

---

## "API Shows as Disconnected"

Check that containers are running:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs api
```

---

# 📖 More Information

See **full-guide.md** for:

- REST API reference
- Advanced configuration
- Deployment options
- Whisper model recommendations
- Troubleshooting and optimization tips