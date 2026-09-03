"""
FastAPI backend for Whisper ASR using faster-whisper.
Optimized for CPU utilization via CTranslate2 int8 quantization.
"""
from __future__ import annotations

import os
import json
import zipfile
import tarfile
import uuid
import logging
import multiprocessing
from pathlib import Path
from typing import Generator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio
import shutil
import tempfile

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Form 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("whisper-api")

# Basic Environment Settings
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "500"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "20"))
MAX_ARCHIVE_FILES = int(os.environ.get("MAX_ARCHIVE_FILES", "100"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
WHISPER_THREADS = int(os.environ.get("WHISPER_THREADS", "0"))
CPU_OPTIMIZED = os.environ.get("CPU_OPTIMIZED", "true").lower() == "true"
USE_VAD = os.environ.get("USE_VAD", "false").lower() == "true"

ALLOWED_EXTENSIONS = {
    '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus',
    '.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv'
}

@dataclass
class HardwareInfo:
    device: str
    cores_available: int
    threads: int
    workers: int
    gpu_count: int
    gpu_names: list[str]
    compute_type: str

def detect_hardware() -> HardwareInfo:
    cores_available = multiprocessing.cpu_count()
    threads = WHISPER_THREADS if WHISPER_THREADS > 0 else cores_available
    threads = max(1, threads)
    workers = min(4, cores_available)
    gpu_count = 0
    gpu_names: list[str] = []
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    except ImportError:
        pass
    
    device = "cuda" if gpu_count > 0 else "cpu"
    
    # Logic for CTranslate2 optimization:
    # CUDA -> float16
    # CPU + Optimized -> int8 (the fastest CPU path)
    # CPU Standard -> int8_float16
    if device == "cuda":
        compute_type = "float16"
    elif CPU_OPTIMIZED:
        compute_type = "int8"
    else:
        compute_type = "int8_float16"
        
    return HardwareInfo(device, cores_available, threads, workers, gpu_count, gpu_names, compute_type)

HARDWARE_INFO = detect_hardware()

# =============================================================================
# Model Management (SINGLE SOURCE OF TRUTH)
# =============================================================================

def _load_models_config() -> dict:
    """Load model configuration from JSON file. Raises FileNotFoundError if not found."""
    possible_paths = [
        Path(__file__).parent / "config" / "models.json",
        Path("/app/config/models.json"),
        Path.cwd() / "config" / "models.json",
    ]
    
    for config_path in possible_paths:
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                logger.info(f"Loaded single source of truth config from: {config_path}")
                return config
    
    error_msg = "CRITICAL: config/models.json not found. This file is the single source of truth for available models."
    logger.critical(error_msg)
    raise FileNotFoundError(error_msg)

# Load config first
MODELS_CONFIG = _load_models_config()

# Derive everything else from the config
DEFAULT_MODEL = MODELS_CONFIG.get("default")
AVAILABLE_MODEL_IDS = {m["id"] for m in MODELS_CONFIG.get("models", [])}

if not DEFAULT_MODEL or not AVAILABLE_MODEL_IDS:
    logger.critical("CRITICAL: config/models.json is malformed. Missing 'default' or 'models' list.")
    raise ValueError("Invalid models.json structure")

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
            
            # PROOF LOGGING: Announces exactly what is happening during load
            logger.info(
                f"🚀 LOADING MODEL: [{model_id}] | "
                f"Device: {HARDWARE_INFO.device} | "
                f"Compute Type: {HARDWARE_INFO.compute_type} (CPU_OPTIMIZED: {CPU_OPTIMIZED}) | "
                f"Threads: {HARDWARE_INFO.threads}"
            )
            
            model = WhisperModel(
                model_id, 
                device=HARDWARE_INFO.device, 
                compute_type=HARDWARE_INFO.compute_type,
                cpu_threads=HARDWARE_INFO.threads, 
                num_workers=HARDWARE_INFO.workers, 
                download_root="/app/models"
            )
            
            logger.info(f"✅ MODEL LOADED: [{model_id}] is now ready for inference.")
            _model_cache[model_id] = model
        return _model_cache[model_id]

def transcribe_single_file(file_path: str, language: str, return_timestamps: bool, model_id: str) -> dict:
    try:
        model = get_whisper_model(model_id)
        language_param = None if language == "auto" else language
        segments, info = model.transcribe(
            file_path, language=language_param, beam_size=5,
            vad_filter=USE_VAD, vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400} if USE_VAD else None
        )
        full_text = []
        timestamped = []
        for s in segments:
            full_text.append(s.text.strip())
            if return_timestamps:
                timestamped.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
        return {
            "filename": os.path.basename(file_path), "text": " ".join(full_text),
            "language": info.language or "unknown", "language_probability": info.language_probability or 0.0,
            "segments": timestamped, "success": True
        }
    except Exception as e:
        return {"filename": os.path.basename(file_path), "success": False, "error": str(e), "text": "", "language": "", "language_probability": 0.0, "segments": []}

# =============================================================================
# API Setup
# =============================================================================

app = FastAPI(title="Whisper ASR API")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS.split(","), allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
executor = ThreadPoolExecutor(max_workers=HARDWARE_INFO.workers)

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    model: str
    device: str
    segments: list
    request_id: str

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
    return {
        "status": "healthy", 
        "device": HARDWARE_INFO.device, 
        "compute_type": HARDWARE_INFO.compute_type,
        "cpu_optimized": CPU_OPTIMIZED,
        "default_model": DEFAULT_MODEL
    }

@app.get("/models")
def list_models():
    return {"models": MODELS_CONFIG["models"], "default": DEFAULT_MODEL}

@app.get("/languages")
def list_languages():
    return {"languages": [{"code": "auto", "name": "Auto Detect"}, {"code": "en", "name": "English"}, {"code": "fi", "name": "Finnish"}]}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False),
    model_id: str | None = Form(None),
):
    request_id = str(uuid.uuid4())[:8]
    valid_model = get_valid_model_id(model_id)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"input{os.path.splitext(file.filename)[1]}"
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, transcribe_single_file, str(tmp_path), language, return_timestamps, valid_model)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return TranscriptionResponse(
            text=result["text"], language=result["language"], language_probability=result["language_probability"],
            model=valid_model, device=HARDWARE_INFO.device, segments=result["segments"], request_id=request_id
        )

@app.post("/transcribe-batch", response_model=BatchResponse)
async def transcribe_batch(
    files: list[UploadFile] = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False),
    model_id: str | None = Form(None),
):
    request_id = str(uuid.uuid4())[:8]
    valid_model = get_valid_model_id(model_id)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tasks = []
        loop = asyncio.get_running_loop()
        for i, file in enumerate(files):
            path = Path(tmp_dir) / f"file_{i}_{os.path.splitext(file.filename)[1]}"
            with open(path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            tasks.append(loop.run_in_executor(executor, transcribe_single_file, str(path), language, return_timestamps, valid_model))
        
        results = await asyncio.gather(*tasks)
        successful = sum(1 for r in results if r["success"])
        
        return BatchResponse(
            total=len(results), successful=successful, failed=len(results)-successful,
            model=valid_model, device=HARDWARE_INFO.device, results=results, request_id=request_id
        )

@app.post("/transcribe-archive")
async def transcribe_archive(
    archive: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False),
    model_id: str | None = Form(None),
):
    request_id = str(uuid.uuid4())[:8]
    valid_model = get_valid_model_id(model_id)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / archive.filename
        with open(archive_path, "wb") as f:
            shutil.copyfileobj(archive.file, f)
        
        extracted_files = []
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as z: z.extractall(tmp_dir)
        elif archive_path.suffix in ('.tar', '.gz', '.tgz'):
            with tarfile.open(archive_path, 'r:*') as t: t.extractall(tmp_dir)
        
        for ext in ALLOWED_EXTENSIONS:
            extracted_files.extend([str(p) for p in Path(tmp_dir).rglob(f"*{ext}")])
            
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(executor, transcribe_single_file, fp, language, return_timestamps, valid_model) for fp in extracted_files]
        results = await asyncio.gather(*tasks)
        
        return {"total": len(results), "successful": sum(1 for r in results if r["success"]), "results": results, "request_id": request_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10001)