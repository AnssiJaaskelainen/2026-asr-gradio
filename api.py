"""
FastAPI backend for Whisper ASR.
Exposed on port 10001.
"""
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
import tempfile
import shutil

# Model configuration
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "medium")
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Global model instance
model = None

app = FastAPI(title="Whisper ASR API", version="1.0.0")

# Enable CORS for Gradio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model():
    global model
    if model is None:
        print(f"Loading Whisper {MODEL_SIZE} on {DEVICE}...")
        model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        print("Model loaded successfully!")
    return model


class TranscriptionRequest(BaseModel):
    language: str = "auto"
    return_timestamps: bool = False


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    model: str
    device: str
    segments: list[TranscriptionSegment]


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_SIZE, "device": DEVICE}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: str = "auto",
    return_timestamps: bool = False
):
    """Transcribe an audio or video file."""
    
    # Validate file type
    allowed_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.mp4', '.mkv', '.avi', '.mov', '.webm'}
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name
    
    try:
        asr_model = load_model()
        beam_size = 5 if return_timestamps else 1
        word_timestamps = True if return_timestamps else False
        
        print(f"Transcribing: {file.filename}")
        segments, info = asr_model.transcribe(
            tmp_path,
            language=language if language != "auto" else None,
            beam_size=beam_size,
            word_timestamps=word_timestamps
        )
        
        segment_list = []
        full_text = ""
        
        for segment in segments:
            full_text += segment.text + " "
            if return_timestamps:
                segment_list.append(TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip()
                ))
        
        return TranscriptionResponse(
            text=full_text.strip(),
            language=info.language,
            language_probability=info.language_probability,
            model=MODEL_SIZE,
            device=DEVICE,
            segments=segment_list
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("Starting Whisper ASR FastAPI Server")
    print(f"Model: {MODEL_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Port: 10001")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=10001)