# api.py
"""
FastAPI backend for Whisper ASR using faster-whisper.
Optimized for archival material: VAD disabled to prevent speech clipping.

Features:
- Persistent storage of transcriptions to /transcripts volume.
- Session-based organization (timestamped folders).
- Static file serving for direct access to transcriptions.
- Batch and Archive (Zip) transcription processing.
- Full export of all transcriptions as a ZIP archive.
"""
from __future__ import annotations

import os
import json
import zipfile
import tarfile
import uuid
import logging
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Generator, AsyncGenerator, List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio
import shutil
import tempfile

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask 
from pydantic import BaseModel

# =============================================================================
# Configuration & Performance Tuning
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("whisper-api")

MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "500"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
TRANSCRIPTS_DIR = os.environ.get("TRANSCRIPTS_DIR", "/transcripts")

WHISPER_THREADS = int(os.environ.get("WHISPER_THREADS", "0"))
CPU_OPTIMIZED = os.environ.get("CPU_OPTIMIZED", "true").lower() == "true"
USE_VAD = False # Forced False for archival quality

ALLOWED_EXTENSIONS = {
    '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus',
    '.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv'
}

@dataclass
class HardwareInfo:
    device: str
    threads: int
    workers: int
    compute_type: str

def detect_hardware() -> HardwareInfo:
    cores_available = multiprocessing.cpu_count()
    threads = WHISPER_THREADS if WHISPER_THREADS > 0 else max(1, cores_available // 2)
    workers = min(4, cores_available // 2)
    
    gpu_count = 0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
    except ImportError:
        pass
    
    device = "cuda" if gpu_count > 0 else "cpu"
    if device == "cuda": compute_type = "float16"
    elif CPU_OPTIMIZED: compute_type = "int8"
    else: compute_type = "int8_float16"
        
    return HardwareInfo(device, threads, workers, compute_type)

HARDWARE_INFO = detect_hardware()

# =============================================================================
# Model Management
# =============================================================================

def _load_models_config() -> dict:
    possible_paths = [
        Path(__file__).parent / "config" / "models.json",
        Path("/app/config/models.json"),
        Path.cwd() / "config" / "models.json",
    ]
    for config_path in possible_paths:
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
    raise FileNotFoundError("CRITICAL: config/models.json not found.")

MODELS_CONFIG = _load_models_config()
DEFAULT_MODEL = MODELS_CONFIG.get("default")
AVAILABLE_MODEL_IDS = {m["id"] for m in MODELS_CONFIG.get("models", [])}

_model_cache: dict = {}
_cache_lock = threading.Lock()

def get_valid_model_id(model_id: str | None) -> str:
    if model_id and model_id in AVAILABLE_MODEL_IDS:
        return model_id
    return DEFAULT_MODEL

def get_whisper_model(model_id: str):
    with _cache_lock:
        if model_id not in _model_cache:
            from faster_whisper import WhisperModel
            logger.info(f"🚀 LOADING MODEL: [{model_id}] | Device: {HARDWARE_INFO.device}")
            model = WhisperModel(
                model_id, 
                device=HARDWARE_INFO.device, 
                compute_type=HARDWARE_INFO.compute_type,
                cpu_threads=HARDWARE_INFO.threads, 
                num_workers=HARDWARE_INFO.workers, 
                download_root="/app/models"
            )
            _model_cache[model_id] = model
        return _model_cache[model_id]

# =============================================================================
# Storage & Formatting Helpers
# =============================================================================

def format_seconds(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"

def format_transcription_text(segments: List[Dict[str, Any]], include_timestamps: bool) -> str:
    if not segments: return ""
    lines = []
    for s in segments:
        text = s.get("text", "").strip()
        if not text: continue
        if include_timestamps:
            lines.append(f"[{format_seconds(s.get('start', 0.0))} -> {format_seconds(s.get('end', 0.0))}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines)

def get_session_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_path = Path(TRANSCRIPTS_DIR) / timestamp
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path

def save_transcription_to_volume(session_dir: Path, original_filename: str, segments: List[Dict[str, Any]], include_timestamps: bool) -> str:
    stem = Path(original_filename).stem
    out_path = session_dir / f"{stem}.txt"
    text = format_transcription_text(segments, include_timestamps)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return str(out_path)

# =============================================================================
# Transcription Logic
# =============================================================================

def transcribe_single_file(file_path: str, language: str, return_timestamps: bool, model_id: str) -> dict:
    try:
        model = get_whisper_model(model_id)
        segments, info = model.transcribe(file_path, language=None if language == "auto" else language, beam_size=1, vad_filter=USE_VAD)
        
        full_text = []
        timestamped = []
        for s in segments:
            full_text.append(s.text.strip())
            timestamped.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
        
        return {
            "filename": os.path.basename(file_path), "text": " ".join(full_text),
            "language": info.language or "unknown", "language_probability": info.language_probability or 0.0,
            "duration": info.duration or 0.0, "segments": timestamped, "success": True
        }
    except Exception as e:
        return {"filename": os.path.basename(file_path), "success": False, "error": str(e)}

def transcribe_streaming(file_path: str, language: str, return_timestamps: bool, model_id: str) -> Generator[dict, None, None]:
    try:
        model = get_whisper_model(model_id)
        segments_gen, info = model.transcribe(file_path, language=None if language == "auto" else language, beam_size=1, vad_filter=USE_VAD)
        yield {"type": "metadata", "duration": info.duration or 0.0, "language": info.language or "unknown", "language_probability": info.language_probability or 0.0}
        full_text = []
        for idx, segment in enumerate(segments_gen):
            text = segment.text.strip()
            full_text.append(text)
            yield {"type": "segment", "start": round(segment.start, 2), "end": round(segment.end, 2), "text": text, "index": idx}
        yield {"type": "done", "text": " ".join(full_text), "success": True, "error": None}
    except Exception as e:
        yield {"type": "done", "text": "", "success": False, "error": str(e)}

# =============================================================================
# API Setup
# =============================================================================

app = FastAPI(title="Whisper ASR API")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS.split(","), allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
executor = ThreadPoolExecutor(max_workers=HARDWARE_INFO.workers)

if os.path.exists(TRANSCRIPTS_DIR):
    app.mount("/static/transcripts", StaticFiles(directory=TRANSCRIPTS_DIR), name="transcripts")
    logger.info(f"📂 Serving transcriptions from {TRANSCRIPTS_DIR} at /static/transcripts")
else:
    logger.warning(f"⚠️ Warning: Transcripts directory {TRANSCRIPTS_DIR} not found. File serving is disabled.")

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    model: str
    device: str
    segments: list
    request_id: str
    file_path: str

class BatchResponse(BaseModel):
    total: int
    successful: int
    failed: int
    model: str
    device: str
    results: list
    request_id: str

@app.get("/health")
def health():
    return {"status": "healthy", "device": HARDWARE_INFO.device, "default_model": DEFAULT_MODEL}

@app.get("/models")
def list_models():
    return {"models": MODELS_CONFIG["models"], "default": DEFAULT_MODEL}

@app.get("/languages")
def list_languages():
    return {"languages": [{"code": "auto", "name": "Auto Detect"}, {"code": "en", "name": "English"}, {"code": "fi", "name": "Finnish"}]}

def _safe_next(gen: Generator):
    try: return next(gen)
    except StopIteration: return None

async def _stream_transcription_events(file_path: str, language: str, return_timestamps: bool, model_id: str, cleanup_func: callable = None) -> AsyncGenerator[str, None]:
    gen = None
    try:
        loop = asyncio.get_running_loop()
        gen = await loop.run_in_executor(executor, lambda: transcribe_streaming(file_path, language, return_timestamps, model_id))
        while True:
            event = await loop.run_in_executor(executor, _safe_next, gen)
            if event is None: break
            yield f"data: {json.dumps(event)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'done', 'success': False, 'error': str(e)})}\n\n"
    finally:
        if cleanup_func: cleanup_func()

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False),
    model_id: str | None = Form(None),
    stream: bool = Form(False),
):
    request_id = str(uuid.uuid4())[:8]
    valid_model = get_valid_model_id(model_id)
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = temp_file.name
    try:
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()
        if stream:
            def cleanup():
                try: os.unlink(temp_path)
                except: pass
            return StreamingResponse(_stream_transcription_events(temp_path, language, return_timestamps, valid_model, cleanup), media_type="text/event-stream")
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(executor, transcribe_single_file, temp_path, language, return_timestamps, valid_model)
            try: os.unlink(temp_path)
            except: pass
            if not result["success"]: raise HTTPException(status_code=500, detail=result.get("error"))
            
            session_dir = get_session_dir()
            saved_path = save_transcription_to_volume(session_dir, file.filename, result["segments"], return_timestamps)
            
            return TranscriptionResponse(
                text=result["text"], language=result["language"], language_probability=result["language_probability"],
                model=valid_model, device=HARDWARE_INFO.device, segments=result["segments"], request_id=request_id, file_path=saved_path
            )
    except Exception as e:
        try: os.unlink(temp_path)
        except: pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe-batch", response_model=BatchResponse)
async def transcribe_batch(
    files: list[UploadFile] = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False),
    model_id: str | None = Form(None),
):
    request_id = str(uuid.uuid4())[:8]
    valid_model = get_valid_model_id(model_id)
    session_dir = get_session_dir()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tasks = []
        loop = asyncio.get_running_loop()
        for i, file in enumerate(files):
            path = Path(tmp_dir) / f"file_{i}_{os.path.splitext(file.filename)[1]}"
            with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
            tasks.append(loop.run_in_executor(executor, transcribe_single_file, str(path), language, return_timestamps, valid_model))
        
        raw_results = await asyncio.gather(*tasks)
        final_results = []
        successful = 0
        
        for i, res in enumerate(raw_results):
            if res["success"]:
                successful += 1
                saved_path = save_transcription_to_volume(session_dir, files[i].filename, res["segments"], return_timestamps)
                res["file_path"] = saved_path
            final_results.append(res)
            
        return BatchResponse(total=len(final_results), successful=successful, failed=len(final_results)-successful, model=valid_model, device=HARDWARE_INFO.device, results=final_results, request_id=request_id)

@app.post("/transcribe-archive")
async def transcribe_archive(
    archive: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False),
    model_id: str | None = Form(None),
):
    request_id = str(uuid.uuid4())[:8]
    valid_model = get_valid_model_id(model_id)
    session_dir = get_session_dir()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / archive.filename
        with open(archive_path, "wb") as f: shutil.copyfileobj(archive.file, f)
        extracted_files = []
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as z: z.extractall(tmp_dir)
        elif archive_path.suffix in ('.tar', '.gz', '.tgz'):
            with tarfile.open(archive_path, 'r:*') as t: t.extractall(tmp_dir)
        
        for ext in ALLOWED_EXTENSIONS:
            extracted_files.extend([str(p) for p in Path(tmp_dir).rglob(f"*{ext}")])
        
        # Validation: Ensure there are files to process
        if not extracted_files:
            raise HTTPException(
                status_code=400, 
                detail="No supported filetypes found inside the uploaded archive file. Please ensure it contains supported audio or video files."
            )
        
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(executor, transcribe_single_file, fp, language, return_timestamps, valid_model) for fp in extracted_files]
        raw_results = await asyncio.gather(*tasks)
        
        final_results = []
        for res in raw_results:
            if res["success"]:
                saved_path = save_transcription_to_volume(session_dir, res["filename"], res["segments"], return_timestamps)
                res["file_path"] = saved_path
            final_results.append(res)
            
        return {"total": len(final_results), "successful": sum(1 for r in final_results if r["success"]), "results": final_results, "request_id": request_id}

# =============================================================================
# History & Archive Endpoints
# =============================================================================

@app.get("/transcripts")
async def list_transcripts():
    if not os.path.exists(TRANSCRIPTS_DIR):
        return {"sessions": []}
    
    sessions = []
    for session_folder in sorted(os.listdir(TRANSCRIPTS_DIR), reverse=True):
        folder_path = Path(TRANSCRIPTS_DIR) / session_folder
        if folder_path.is_dir():
            files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
            sessions.append({
                "session_id": session_folder,
                "files": files,
                "created_at": session_folder
            })
    return {"sessions": sessions}

@app.get("/transcripts/archive")
async def download_all_transcripts():
    if not os.path.exists(TRANSCRIPTS_DIR):
        raise HTTPException(status_code=404, detail="No transcriptions found to archive.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        zip_path = tmp_zip.name
        
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(TRANSCRIPTS_DIR):
                for file in files:
                    file_full_path = Path(root) / file
                    archive_name = file_full_path.relative_to(TRANSCRIPTS_DIR)
                    zipf.write(file_full_path, archive_name)
        
        return FileResponse(
            path=zip_path, 
            filename="all_transcriptions.zip", 
            media_type="application/zip",
            background=BackgroundTask(os.unlink, zip_path)
        )
    except Exception as e:
        if os.path.exists(zip_path): os.unlink(zip_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10001)