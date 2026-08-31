"""
Gradio frontend for Whisper ASR.
Connects to FastAPI backend on port 10001.
Exposed on port 10002.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Generator, Optional

import gradio as gr
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

API_URL = os.environ.get("API_URL", "http://localhost:10001")
API_DOCS_URL = os.environ.get("API_DOCS_URL", "http://localhost:10001/docs")

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "medium")
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Cache for languages and models (refreshed on load)
_languages_cache: Optional[list[dict]] = None
_models_cache: Optional[list[dict]] = None
_cache_lock = threading.Lock()


def get_languages() -> list[dict]:
    """Fetch supported languages from API (with caching)."""
    global _languages_cache
    
    if _languages_cache is not None:
        return _languages_cache
    
    try:
        response = requests.get(f"{API_URL}/languages", timeout=10)
        response.raise_for_status()
        data = response.json()
        _languages_cache = data.get("languages", [])
        return _languages_cache
    except RequestException as e:
        print(f"Warning: Could not fetch languages from API: {e}")
        return [
            {"code": "auto", "name": "Auto Detect"},
            {"code": "en", "name": "English"},
        ]


def get_models() -> list[dict]:
    """Fetch available models from API (with caching)."""
    global _models_cache
    
    if _models_cache is not None:
        return _models_cache
    
    try:
        response = requests.get(f"{API_URL}/models", timeout=10)
        response.raise_for_status()
        data = response.json()
        _models_cache = data.get("models", [])
        return _models_cache
    except RequestException as e:
        print(f"Warning: Could not fetch models from API: {e}")
        return [
            {"id": "medium", "name": "Medium (769M)", "description": "Whisper medium == Balances speed and accuracy"}
        ]


def get_language_choices() -> list[str]:
    """Get language choices for dropdown."""
    languages = get_languages()
    return [lang["code"] for lang in languages]


def get_model_choices() -> list[str]:
    """Get model choices for dropdown (using description as display)."""
    models = get_models()
    return [m["description"] for m in models]


def get_model_id_map() -> dict[str, str]:
    """Get mapping from description to model id."""
    models = get_models()
    return {m["description"]: m["id"] for m in models}


@contextmanager
def managed_file(file_path: str) -> Generator[None, None, None]:
    """Context manager for safe file handling."""
    f = None
    try:
        f = open(file_path, "rb")
        yield
    finally:
        if f:
            f.close()


def transcribe_files(
    files: list[str],
    language: str,
    model_description: str,
    return_timestamps: bool
) -> tuple[str, str]:
    """Transcribe single or multiple audio/video files."""
    if not files:
        return "", "No files provided"
    
    valid_paths = [p for p in files if p]
    if not valid_paths:
        return "", "No valid files"
    
    # Convert description to model id
    model_id_map = get_model_id_map()
    model_id = model_id_map.get(model_description, "medium")
    
    try:
        files_data: list[tuple] = []
        try:
            for path in valid_paths:
                f = open(path, "rb")
                files_data.append(("files", (os.path.basename(path), f)))
            
            response = requests.post(
                f"{API_URL}/transcribe-batch",
                files=files_data,
                data={
                    "language": language,
                    "model": model_id,
                    "return_timestamps": return_timestamps
                },
                timeout=600
            )
        finally:
            for _, (_, f) in files_data:
                try:
                    f.close()
                except Exception:
                    pass
        
        if response.status_code != 200:
            detail = response.json().get("detail", "Unknown error")
            return "", f"Error: {detail}"
        
        result = response.json()
        
        output_lines: list[str] = [
            f"Model: {result['model']} | Total: {result['total']} | "
            f"Success: {result['successful']} | Failed: {result['failed']}\n"
        ]
        
        for item in result["results"]:
            output_lines.append(f"\n--- {item['filename']} ---")
            
            if item["success"]:
                output_lines.append(f"[{item['language']} ({item['language_probability']:.1%})]")
                output_lines.append(item["text"])
                
                if return_timestamps:
                    for seg in item["segments"]:
                        start = seg.get("start", 0)
                        end = seg.get("end", 0)
                        text = seg.get("text", "")
                        output_lines.append(f"  [{start:.2f}s - {end:.2f}s] {text}")
            else:
                output_lines.append(f"ERROR: {item.get('error', 'Unknown error')}")
        
        return "\n".join(output_lines), ""
    
    except ConnectionError:
        return "", "Error: Cannot connect to API"
    except Timeout:
        return "", "Error: Request timed out"
    except RequestException as e:
        return "", f"Error: {str(e)}"


def transcribe_archive(
    archive_path: str,
    language: str,
    model_description: str,
    return_timestamps: bool
) -> tuple[str, str]:
    """Upload and transcribe files from archive."""
    if not archive_path:
        return "", "No archive provided"
    
    # Convert description to model id
    model_id_map = get_model_id_map()
    model_id = model_id_map.get(model_description, "medium")
    
    try:
        with managed_file(archive_path):
            response = requests.post(
                f"{API_URL}/transcribe-archive",
                files={"archive": (os.path.basename(archive_path), open(archive_path, "rb"))},
                data={
                    "language": language,
                    "model": model_id,
                    "return_timestamps": return_timestamps
                },
                timeout=600
            )
        
        if response.status_code != 200:
            detail = response.json().get("detail", "Unknown error")
            return "", f"Error: {detail}"
        
        result = response.json()
        
        output_lines: list[str] = [
            f"Archive: {result['archive']} | Model: {result['model']} | "
            f"Total: {result['total']} | Success: {result['successful']} | Failed: {result['failed']}\n"
        ]
        
        for item in result["results"]:
            output_lines.append(f"\n--- {item['filename']} ---")
            
            if item["success"]:
                output_lines.append(f"[{item['language']} ({item['language_probability']:.1%})]")
                output_lines.append(item["text"])
                
                if return_timestamps:
                    for seg in item["segments"]:
                        start = seg.get("start", 0)
                        end = seg.get("end", 0)
                        text = seg.get("text", "")
                        output_lines.append(f"  [{start:.2f}s - {end:.2f}s] {text}")
            else:
                output_lines.append(f"ERROR: {item.get('error', 'Unknown error')}")
        
        return "\n".join(output_lines), ""
    
    except RequestException as e:
        return "", f"Error: {str(e)}"


LANGUAGE_CHOICES = get_language_choices()
MODEL_CHOICES = get_model_choices()
MODEL_ID_MAP = get_model_id_map()
DEFAULT_LANGUAGE = "auto"
DEFAULT_MODEL = MODEL_CHOICES[0] if MODEL_CHOICES else "Whisper medium == Balances speed and accuracy"


def build_ui() -> gr.Blocks:
    """Build the Gradio UI."""
    with gr.Blocks(title="Whisper ASR Transcription") as demo:
        gr.Markdown("# Whisper ASR Transcription")
        gr.Markdown(
            f"**Default Model:** {MODEL_SIZE} | **Device:** {DEVICE} | "
            f"**API:** [{API_DOCS_URL}]({API_DOCS_URL})"
        )
        
        with gr.Tab("Audio/Video Files"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="Upload Audio or Video Files",
                        file_count="multiple",
                        file_types=["audio", "video"]
                    )
                    model_dropdown = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value=DEFAULT_MODEL,
                        label="Model",
                        allow_custom_value=False
                    )
                    language = gr.Dropdown(
                        choices=LANGUAGE_CHOICES,
                        value=DEFAULT_LANGUAGE,
                        label="Language",
                        allow_custom_value=False
                    )
                    timestamps = gr.Checkbox(
                        label="Include Timestamps",
                        value=False
                    )
                    transcribe_btn = gr.Button(
                        "Transcribe",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    output = gr.Textbox(
                        label="Transcription Results",
                        lines=20
                    )
                    status = gr.Textbox(
                        label="Status",
                        lines=2,
                        interactive=False
                    )
            
            transcribe_btn.click(
                fn=transcribe_files,
                inputs=[file_input, language, model_dropdown, timestamps],
                outputs=[output, status]
            )
        
        with gr.Tab("Archive (ZIP/TAR)"):
            with gr.Row():
                with gr.Column(scale=1):
                    archive_input = gr.File(
                        label="Upload ZIP, TAR, or TAR.GZ",
                        file_types=[".zip", ".tar", ".gz", ".tgz", ".tar.gz"]
                    )
                    model_archive = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value=DEFAULT_MODEL,
                        label="Model",
                        allow_custom_value=False
                    )
                    language_archive = gr.Dropdown(
                        choices=LANGUAGE_CHOICES,
                        value=DEFAULT_LANGUAGE,
                        label="Language",
                        allow_custom_value=False
                    )
                    timestamps_archive = gr.Checkbox(
                        label="Include Timestamps",
                        value=False
                    )
                    transcribe_archive_btn = gr.Button(
                        "Extract & Transcribe",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    archive_output = gr.Textbox(
                        label="Archive Results",
                        lines=20
                    )
                    archive_status = gr.Textbox(
                        label="Status",
                        lines=2,
                        interactive=False
                    )
            
            transcribe_archive_btn.click(
                fn=transcribe_archive,
                inputs=[archive_input, language_archive, model_archive, timestamps_archive],
                outputs=[archive_output, archive_status]
            )
    
    return demo


demo = build_ui()


if __name__ == "__main__":
    print("=" * 50)
    print(f"Whisper ASR Gradio | API: {API_URL} | Port: 10002")
    print(f"Available Models: {MODEL_CHOICES}")
    print("=" * 50)
    demo.launch(
        server_name="0.0.0.0",
        server_port=10002,
        share=False
    )