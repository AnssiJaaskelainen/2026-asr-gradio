"""
FastAPI backend for Whisper ASR.
Exposed on port 10001.
"""
from __future__ import annotations

import os
import json
import zipfile
import tarfile
import uuid
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator
from enum import Enum

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
import tempfile
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("whisper-api")

# Configuration
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "500"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "20"))
MAX_ARCHIVE_FILES = int(os.environ.get("MAX_ARCHIVE_FILES", "100"))

# Thread pool for CPU-bound transcription
executor = ThreadPoolExecutor(max_workers=4)

# Lazy-loaded models cache
_models: dict[str, WhisperModel] = {}

# Load models from config file
def load_models_config() -> dict:
    """Load available models from JSON config file."""
    config_path = os.environ.get("MODELS_CONFIG", "/app/config/models.json")
    default_config = {
        "models": [
            {"id": "tiny", "name": "Tiny (39M)", "description": "Whisper tiny == Fastest but not that accurate"},
            {"id": "medium", "name": "Medium (769M)", "description": "Whisper medium == Balances speed and accuracy"},
            {"id": "large-v3", "name": "Large v3 (1550M)", "description": "Whisper large-v3 == Best accuracy but slowest"}
        ],
        "default": "medium"
    }
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            logger.info(f"Loaded models config from {config_path}")
            return config
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return default_config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return default_config

# Load config at startup
MODELS_CONFIG = load_models_config()
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", MODELS_CONFIG.get("default", "medium"))

# Create Enum with description as display name
# Map: sanitized_description -> model_id
MODEL_DESCRIPTIONS = [m["description"] for m in MODELS_CONFIG.get("models", [])]
MODEL_DESC_TO_ID = {m["description"]: m["id"] for m in MODELS_CONFIG.get("models", [])}

def sanitize_enum_name(name: str) -> str:
    """Convert description to valid enum name."""
    return name.replace(" ", "_").replace("==", "eq").replace("-", "_").replace("(", "").replace(")", "")[:50]

if MODEL_DESCRIPTIONS:
    enum_members = {sanitize_enum_name(desc): desc for desc in MODEL_DESCRIPTIONS}
    ModelEnum = Enum("ModelEnum", enum_members)
else:
    ModelEnum = Enum("ModelEnum", {"Balanced": "Whisper medium == Balances speed and accuracy"})

app = FastAPI(
    title="Whisper ASR API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
_cors_origins = os.environ.get("CORS_ORIGINS", "")
if _cors_origins:
    _allowed_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]
else:
    _allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS: set[str] = {
    '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac',
    '.mp4', '.mkv', '.avi', '.mov', '.webm', '.wma', '.aiff'
}


def validate_device() -> tuple[str, str]:
    """Validate and return proper device configuration."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    
    if cuda_visible and cuda_visible != "-1":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda", "float16"
        except ImportError:
            pass
    
    return "cpu", "int8"


def load_model(model_size: str) -> WhisperModel:
    """Load Whisper model (cached by size)."""
    if model_size not in _models:
        device, compute_type = validate_device()
        logger.info(f"Loading Whisper {model_size} on {device} (compute={compute_type})...")
        
        try:
            _models[model_size] = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            logger.info(f"Model {model_size} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_size}: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    return _models[model_size]


def is_audio_or_video(filename: str) -> bool:
    """Check if file extension is allowed."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def is_safe_path(path: Path, base_dir: Path) -> bool:
    """Check if path is safely within base directory (prevents zip slip)."""
    try:
        resolved = path.resolve()
        base_resolved = base_dir.resolve()
        return resolved == base_dir or base_dir in resolved.parents or resolved.parent == base_dir
    except (ValueError, OSError):
        return False


def extract_files_from_archive(archive_path: str, extract_dir: str) -> list[str]:
    """Extract audio/video files from archive with security checks."""
    extracted: list[str] = []
    extract_path = Path(extract_dir)
    
    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                for name in zf.namelist():
                    if name.endswith('/') or name.startswith('.'):
                        continue
                    if name.startswith('/') or '..' in name:
                        logger.warning(f"Skipping unsafe path in archive: {name}")
                        continue
                    if is_audio_or_video(name):
                        target = extract_path / name
                        if is_safe_path(target, extract_path):
                            extracted.append(str(target))
                
                zf.extractall(extract_dir)
        
        elif archive_path.endswith(('.tar', '.tgz')) or 'tar.gz' in archive_path:
            with tarfile.open(archive_path, 'r:*') as tf:
                for member in tf.getmembers():
                    if member.name.startswith('/') or '..' in member.name:
                        logger.warning(f"Skipping unsafe path in archive: {member.name}")
                        continue
                    if member.isfile() and is_audio_or_video(member.name):
                        target = extract_path / member.name
                        if is_safe_path(target, extract_path):
                            extracted.append(str(target))
                
                tf.extractall(extract_dir)
        
        elif archive_path.endswith('.gz'):
            import gzip
            basename = os.path.basename(archive_path).replace('.gz', '')
            if is_audio_or_video(basename):
                output_path = extract_path / basename
                if is_safe_path(output_path, extract_path):
                    with gzip.open(archive_path, 'rb') as f_in:
                        with open(output_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    extracted.append(str(output_path))
    
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        raise
    
    return extracted


@contextmanager
def temporary_directory() -> Generator[Path, None, None]:
    """Context manager for safe temp directory cleanup."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="whisper_"))
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def transcribe_single_file(
    file_path: str,
    language: str,
    return_timestamps: bool,
    model_size: str
) -> dict:
    """Transcribe a single audio file."""
    asr_model = load_model(model_size)
    
    segments_gen, info = asr_model.transcribe(
        file_path,
        language=language if language != "auto" else None,
        beam_size=5 if return_timestamps else 1,
        word_timestamps=return_timestamps
    )
    
    segment_list = list(segments_gen)
    
    segment_data = [
        {
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": s.text.strip()
        }
        for s in segment_list
    ]
    
    full_text = " ".join(s.text for s in segment_list).strip()
    
    return {
        "filename": os.path.basename(file_path),
        "text": full_text,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "segments": segment_data,
        "success": True,
        "error": None
    }


class TranscriptionResponse(BaseModel):
    """Response model for single file transcription."""
    text: str
    language: str
    language_probability: float
    model: str
    device: str
    segments: list
    request_id: Optional[str] = None


class BatchResponse(BaseModel):
    """Response model for batch transcription."""
    total: int
    successful: int
    failed: int
    model: str
    device: str
    results: list
    request_id: Optional[str] = None


@app.get("/")
def root() -> dict:
    """Root endpoint with service info."""
    device, _ = validate_device()
    return {
        "service": "Whisper ASR API",
        "version": "1.0.0",
        "default_model": DEFAULT_MODEL,
        "available_models": MODELS_CONFIG.get("models", []),
        "device": device,
        "status": "ok"
    }


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    device, _ = validate_device()
    return {
        "status": "healthy",
        "models_loaded": list(_models.keys()),
        "default_model": DEFAULT_MODEL,
        "device": device
    }


@app.get("/models")
def list_models() -> dict:
    """List available Whisper models from config."""
    return {
        "models": MODELS_CONFIG.get("models", []),
        "default": DEFAULT_MODEL,
        "loaded": list(_models.keys())
    }


@app.get("/languages")
def list_languages() -> dict:
    """List supported languages for transcription."""
    return {
        "languages": [
            {"code": "auto", "name": "Auto Detect"},
            {"code": "en", "name": "English"},
            {"code": "fi", "name": "Finnish"},
            {"code": "sv", "name": "Swedish"},
            {"code": "de", "name": "German"},
            {"code": "fr", "name": "French"},
            {"code": "es", "name": "Spanish"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "ru", "name": "Russian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "it", "name": "Italian"},
            {"code": "pl", "name": "Polish"},
            {"code": "nl", "name": "Dutch"},
            {"code": "ar", "name": "Arabic"},
        ]
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    request: Request,
    file: UploadFile = File(..., description="Audio or video file to transcribe"),
    language: str = Query(default="auto", description="Language code (e.g., 'en', 'fi') or 'auto' for detection"),
    return_timestamps: bool = Query(default=False, description="Include word-level timestamps"),
    model: ModelEnum = Query(default=None, description="Whisper model to use")
) -> TranscriptionResponse:
    """Transcribe a single audio/video file."""
    device, _ = validate_device()
    request_id = str(uuid.uuid4())[:8]
    
    # Convert enum value (description) to model id
    model_size = MODEL_DESC_TO_ID.get(model.value, DEFAULT_MODEL) if model else DEFAULT_MODEL
    
    logger.info(f"[{request_id}] Processing file: {file.filename} with model: {model_size}")
    
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"[{request_id}] Unsupported file type: {ext}")
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    
    with temporary_directory() as tmp_dir:
        tmp_path = tmp_dir / f"audio{ext}"
        
        try:
            with open(tmp_path, 'wb') as f:
                shutil.copyfileobj(file.file, f)
            
            file_size = tmp_path.stat().st_size
            if file_size > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
                )
            
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                executor,
                transcribe_single_file,
                str(tmp_path),
                language,
                return_timestamps,
                model_size
            )
            
            result["request_id"] = request_id
            logger.info(f"[{request_id}] Completed: {result['language']} ({file_size / 1024 / 1024:.1f}MB)")
            
            return TranscriptionResponse(
                text=result["text"],
                language=result["language"],
                language_probability=result["language_probability"],
                model=model_size,
                device=device,
                segments=result["segments"],
                request_id=request_id
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe-batch", response_model=BatchResponse)
async def transcribe_batch(
    files: list[UploadFile] = File(..., description="Multiple audio or video files"),
    language: str = Query(default="auto", description="Language code or 'auto'"),
    return_timestamps: bool = Query(default=False, description="Include word-level timestamps"),
    model: ModelEnum = Query(default=None, description="Whisper model to use")
) -> BatchResponse:
    """Transcribe multiple audio/video files in batch."""
    device, _ = validate_device()
    request_id = str(uuid.uuid4())[:8]
    
    # Convert enum value (description) to model id
    model_size = MODEL_DESC_TO_ID.get(model.value, DEFAULT_MODEL) if model else DEFAULT_MODEL
    
    logger.info(f"[{request_id}] Batch processing {len(files)} files with model: {model_size}")
    
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_BATCH_SIZE} files per batch"
        )
    
    with temporary_directory() as tmp_dir:
        file_paths: list[str] = []
        
        try:
            for i, file in enumerate(files):
                ext = os.path.splitext(file.filename or "")[1].lower()
                if ext not in ALLOWED_EXTENSIONS:
                    logger.warning(f"[{request_id}] Skipping unsupported: {file.filename}")
                    continue
                
                safe_name = os.path.basename(file.filename) or f"file_{i}"
                tmp_path = tmp_dir / safe_name
                
                with open(tmp_path, 'wb') as f:
                    shutil.copyfileobj(file.file, f)
                
                file_size = tmp_path.stat().st_size
                if file_size > MAX_FILE_SIZE_BYTES:
                    logger.warning(f"[{request_id}] Skipping oversized file: {file.filename}")
                    continue
                
                file_paths.append(str(tmp_path))
            
            if not file_paths:
                raise HTTPException(status_code=400, detail="No valid files found")
            
            loop = asyncio.get_running_loop()
            tasks = [
                loop.run_in_executor(
                    executor,
                    transcribe_single_file,
                    fp,
                    language,
                    return_timestamps,
                    model_size
                )
                for fp in file_paths
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            batch_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    batch_results.append({
                        "filename": os.path.basename(file_paths[i]),
                        "text": "",
                        "language": "",
                        "language_probability": 0.0,
                        "segments": [],
                        "success": False,
                        "error": str(result)
                    })
                    logger.error(f"[{request_id}] Failed {file_paths[i]}: {result}")
                else:
                    batch_results.append(result)
            
            successful = sum(1 for r in batch_results if r["success"])
            failed = len(batch_results) - successful
            
            logger.info(f"[{request_id}] Batch complete: {successful} succeeded, {failed} failed")
            
            return BatchResponse(
                total=len(file_paths),
                successful=successful,
                failed=failed,
                model=model_size,
                device=device,
                results=batch_results,
                request_id=request_id
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Batch error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe-archive")
async def transcribe_archive(
    archive: UploadFile = File(..., description="ZIP, TAR, or TAR.GZ archive containing audio/video files"),
    language: str = Query(default="auto", description="Language code or 'auto'"),
    return_timestamps: bool = Query(default=False, description="Include word-level timestamps"),
    model: ModelEnum = Query(default=None, description="Whisper model to use")
) -> dict:
    """Upload and extract files from archive, then transcribe."""
    device, _ = validate_device()
    request_id = str(uuid.uuid4())[:8]
    filename = archive.filename or "archive"
    ext = os.path.splitext(filename)[1].lower()
    
    # Convert enum value (description) to model id
    model_size = MODEL_DESC_TO_ID.get(model.value, DEFAULT_MODEL) if model else DEFAULT_MODEL
    
    logger.info(f"[{request_id}] Processing archive: {filename} with model: {model_size}")
    
    with temporary_directory() as tmp_dir:
        try:
            if ext in ALLOWED_EXTENSIONS:
                tmp_path = tmp_dir / filename
                with open(tmp_path, 'wb') as f:
                    shutil.copyfileobj(archive.file, f)
                file_paths = [str(tmp_path)]
            else:
                archive_path = tmp_dir / filename
                with open(archive_path, 'wb') as f:
                    shutil.copyfileobj(archive.file, f)
                
                file_paths = extract_files_from_archive(str(archive_path), str(tmp_dir))
            
            if not file_paths:
                raise HTTPException(
                    status_code=400,
                    detail="No valid audio/video files found in archive"
                )
            
            if len(file_paths) > MAX_ARCHIVE_FILES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Maximum {MAX_ARCHIVE_FILES} files per archive"
                )
            
            loop = asyncio.get_running_loop()
            tasks = [
                loop.run_in_executor(
                    executor,
                    transcribe_single_file,
                    fp,
                    language,
                    return_timestamps,
                    model_size
                )
                for fp in file_paths
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            batch_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    batch_results.append({
                        "filename": os.path.basename(file_paths[i]),
                        "text": "",
                        "language": "",
                        "language_probability": 0.0,
                        "segments": [],
                        "success": False,
                        "error": str(result)
                    })
                else:
                    batch_results.append(result)
            
            successful = sum(1 for r in batch_results if r["success"])
            failed = len(batch_results) - successful
            
            logger.info(f"[{request_id}] Archive complete: {successful}/{len(file_paths)} succeeded")
            
            return {
                "archive": filename,
                "total": len(file_paths),
                "successful": successful,
                "failed": failed,
                "model": model_size,
                "device": device,
                "results": batch_results,
                "request_id": request_id
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Archive error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up on shutdown."""
    logger.info("Shutting down, cleaning up resources...")
    executor.shutdown(wait=True)


if __name__ == "__main__":
    import uvicorn
    
    device, _ = validate_device()
    
    logger.info("=" * 50)
    logger.info("Starting Whisper ASR FastAPI Server")
    logger.info(f"Default Model: {DEFAULT_MODEL} | Device: {device} | Port: 10001")
    logger.info(f"Available Models: {[m['id'] for m in MODELS_CONFIG.get('models', [])]}")
    logger.info(f"Max file size: {MAX_FILE_SIZE_MB}MB")
    logger.info("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=10001,
        log_level="info",
        access_log=True
    )