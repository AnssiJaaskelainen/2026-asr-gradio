# app.py
"""
Gradio Web UI for Whisper ASR API.
Saves all transcriptions to a persistent volume /transcripts/[timestamp]/
"""
from __future__ import annotations

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import gradio as gr
import httpx
import json

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("whisper-gradio")

API_BASE_URL = os.environ.get("API_URL", "http://api:10001")
API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "300"))
TRANSCRIPTS_VOLUME_DIR = os.environ.get("TRANSCRIPTS_DIR", "/transcripts")

class WhisperAPIClient:
    def __init__(self, base_url: str = API_BASE_URL, timeout: int = API_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, follow_redirects=True)
        return self._client

    def health_check(self) -> bool:
        try:
            response = self.client.get("/health")
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def get_models(self) -> List[dict]:
        try:
            response = self.client.get("/models")
            response.raise_for_status()
            return response.json().get("models", [])
        except httpx.HTTPError:
            return []

    def get_languages(self) -> List[dict]:
        try:
            response = self.client.get("/languages")
            response.raise_for_status()
            return response.json().get("languages", [])
        except httpx.HTTPError:
            return []

    def transcribe(self, file_path: str, language: str = "auto", return_timestamps: bool = False, model: Optional[str] = None) -> dict:
        with open(file_path, "rb") as f:
            parts = [
                ("file", (Path(file_path).name, f)),
                ("language", (None, language)),
                ("return_timestamps", (None, str(return_timestamps).lower())),
            ]
            if model:
                parts.append(("model_id", (None, model)))
            response = self.client.post("/transcribe", files=parts)
            response.raise_for_status()
            return response.json()

    def transcribe_streaming(self, file_path: str, language: str = "auto", return_timestamps: bool = False, model: Optional[str] = None):
        with open(file_path, "rb") as f:
            parts = [
                ("file", (Path(file_path).name, f)),
                ("language", (None, language)),
                ("return_timestamps", (None, str(return_timestamps).lower())),
                ("stream", (None, "true")),
            ]
            if model:
                parts.append(("model_id", (None, model)))
            
            with self.client.stream("POST", "/transcribe", files=parts, timeout=None) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or line == ':': 
                        continue
                    if line.startswith("data: "):
                        try:
                            data_str = line[6:].strip()
                            if data_str: 
                                yield json.loads(data_str)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue

    def transcribe_archive(self, archive_path: str, language: str = "auto", return_timestamps: bool = False, model: Optional[str] = None) -> dict:
        with open(archive_path, "rb") as f:
            parts = [
                ("archive", (Path(archive_path).name, f)),
                ("language", (None, language)),
                ("return_timestamps", (None, str(return_timestamps).lower())),
            ]
            if model:
                parts.append(("model_id", (None, model)))
            response = self.client.post("/transcribe-archive", files=parts)
            response.raise_for_status()
            return response.json()

api_client = WhisperAPIClient()

# =============================================================================
# File System & Formatting Utilities
# =============================================================================

def format_seconds(seconds: float) -> str:
    """Converts seconds to MM:SS.ms format."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"

def format_transcription_text(segments: List[Dict[str, Any]], include_timestamps: bool) -> str:
    """
    Converts a list of segments into a single string.
    If include_timestamps is True, it adds [00:00.00 -> 00:05.00] prefix.
    """
    if not segments:
        return ""
    
    lines = []
    for s in segments:
        text = s.get("text", "").strip()
        if not text: continue
        
        if include_timestamps:
            start = format_seconds(s.get("start", 0.0))
            end = format_seconds(s.get("end", 0.0))
            lines.append(f"[{start} -> {end}] {text}")
        else:
            lines.append(text)
            
    return "\n".join(lines)

def get_session_dir() -> Path:
    """Creates and returns a timestamped directory within the transcripts volume."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_path = Path(TRANSCRIPTS_VOLUME_DIR) / timestamp
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path

def save_text_to_volume(session_dir: Path, original_filename: str, text: str) -> str:
    """Saves text to a file using the original filename and returns the absolute path."""
    stem = Path(original_filename).stem
    out_path = session_dir / f"{stem}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return str(out_path)

# =============================================================================
# UI Helpers
# =============================================================================

def create_model_choices():
    models = api_client.get_models()
    return [(f"{m.get('name', m['id'])} ({m['id']})", m["id"]) for m in models]

def create_language_choices():
    langs = api_client.get_languages()
    return [(l["name"], l["code"]) for l in langs]

def format_duration(seconds: float) -> str:
    if seconds <= 0: return "Unknown"
    if seconds < 60: return f"{seconds:.1f}s"
    elif seconds < 3600: return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else: return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {int(seconds % 60)}s"

# =============================================================================
# Core Handlers
# =============================================================================

def handle_single_transcription(file, lang, timestamps, model, live_updates, progress=gr.Progress()):
    if not file: raise gr.Error("Please upload a file")
    
    session_dir = get_session_dir()
    original_filename = Path(file.name).name
    
    if live_updates:
        all_segments = []
        full_text_accumulator = []
        total_duration = 0.0
        detected_language = "unknown"
        
        try:
            for event in api_client.transcribe_streaming(file.name, lang, timestamps, model):
                event_type = event.get("type")
                if event_type == "metadata":
                    total_duration = event.get("duration", 0.0)
                    detected_language = event.get("language", "unknown")
                    yield (f"*Initializing...*", "🔄 Starting...", f"{detected_language}", format_duration(total_duration), gr.update())
                elif event_type == "segment":
                    text = event.get("text", "").strip()
                    if not text: continue
                    
                    segment_data = {"start": event.get("start", 0.0), "end": event.get("end", 0.0), "text": text}
                    all_segments.append(segment_data)
                    full_text_accumulator.append(text)
                    
                    # FIX: format the display text in real-time if timestamps are enabled
                    current_display_text = format_transcription_text(all_segments, timestamps) if timestamps else " ".join(full_text_accumulator)
                    
                    seg_end = event.get("end", 0.0)
                    if total_duration > 0: progress(min(seg_end / total_duration, 0.99))
                    yield (current_display_text + "\n\n*Transcribing...*", f"🔄 {seg_end:.1f}s / {total_duration:.1f}s", gr.update(), gr.update(), gr.update())
                elif event_type == "done":
                    final_text = format_transcription_text(all_segments, timestamps) if timestamps else " ".join(full_text_accumulator)
                    file_path = save_text_to_volume(session_dir, original_filename, final_text)
                    yield (f"**✓ Complete**\n\n{final_text[:1000]}...", "✓ Done", gr.update(), gr.update(), file_path)
        except Exception as e:
            yield (f"❌ Error: {str(e)}", "❌ Failed", gr.update(), gr.update(), gr.update())
    else:
        result = api_client.transcribe(file.name, lang, timestamps, model)
        final_text = format_transcription_text(result.get("segments", []), timestamps) if timestamps else result.get("text", "")
        file_path = save_text_to_volume(session_dir, original_filename, final_text)
        return (final_text[:2000], "✓ Done", f"{result.get('language')}", format_duration(result.get('duration', 0)), file_path)

def handle_batch_transcription(files, lang, timestamps, model, progress=gr.Progress()):
    if not files: raise gr.Error("Please upload one or more files")
    
    file_list = files if isinstance(files, list) else [files]
    total_files = len(file_list)
    all_results_summary = []
    download_files = []
    session_dir = get_session_dir()
    
    yield "*Preparing batch...*", "🔄 Queueing...", gr.update()
    
    for i, file in enumerate(file_list):
        filename = Path(file.name).name
        progress((i + 1) / total_files, f"Processing {i+1}/{total_files}...")
        yield "\n\n".join(all_results_summary), f"🔄 Processing: {filename}...", gr.update()
        
        try:
            res = api_client.transcribe(file.name, lang, timestamps, model)
            final_text = format_transcription_text(res.get("segments", []), timestamps) if timestamps else res.get("text", "")
            duration_str = format_duration(res.get("duration", 0))
            
            if "error" in res:
                all_results_summary.append(f"**{filename}** ❌ Error: {res['error']}")
            else:
                all_results_summary.append(f"**{filename}** ({res.get('language')} | {duration_str}): {res.get('text', '')[:100]}...")
                download_files.append(save_text_to_volume(session_dir, filename, final_text))
            
            yield "\n\n".join(all_results_summary), f"✓ Finished: {filename} | {i+1}/{total_files}", gr.update()
        except Exception as e:
            all_results_summary.append(f"**{filename}** ❌ System Error: {str(e)}")
            yield "\n\n".join(all_results_summary), f"⚠️ Error on file {i+1}", gr.update()
    
    yield "\n\n".join(all_results_summary), f"✓ Batch complete", download_files

def handle_archive_transcription(archive, lang, timestamps, model):
    if not archive: raise gr.Error("Please upload an archive")
    
    session_dir = get_session_dir()
    result = api_client.transcribe_archive(archive.name, lang, timestamps, model)
    results_list = result.get("results", [])
    
    download_files = []
    summaries = []
    
    for r in results_list:
        fname = r.get("filename", "unknown")
        final_text = format_transcription_text(r.get("segments", []), timestamps) if timestamps else r.get("text", "")
        summaries.append(f"**{fname}**: {r.get('text', '')[:100]}...")
        download_files.append(save_text_to_volume(session_dir, fname, final_text))
        
    return "\n\n".join(summaries), f"✓ Processed {len(results_list)} files", download_files

# =============================================================================
# Interface Build
# =============================================================================

def build_interface():
    api_healthy = api_client.health_check()
    m_choices = create_model_choices()
    l_choices = create_language_choices()
    default_model_val = m_choices[0][1] if m_choices else None
    all_media = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"]

    with gr.Blocks() as interface:
        gr.Markdown("# 🎙️ Whisper ASR Interface")
        gr.Markdown("🟢 API Connected" if api_healthy else "🔴 API Disconnected")
        
        with gr.Tabs():
            with gr.TabItem("📝 Single Transcribe"):
                with gr.Row():
                    with gr.Column(scale=1):
                        f_in = gr.File(label="File", file_types=all_media)
                        l_in = gr.Dropdown(choices=l_choices, value="auto", label="Language")
                        m_in = gr.Dropdown(choices=m_choices, value=default_model_val, label="Model")
                        t_in = gr.Checkbox(label="Timestamps", value=False)
                        live_in = gr.Checkbox(label="⚡ Live updates", value=True)
                        btn = gr.Button("⚡ Transcribe", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        with gr.Row():
                            d_out = gr.Textbox(label="Duration", interactive=False)
                            l_out = gr.Textbox(label="Detected Language", interactive=False)
                        t_out = gr.Markdown("*Upload a file to begin...*")
                        s_out = gr.Textbox(label="Status", interactive=False)
                        f_out = gr.File(label="Download Transcription", file_count="single")
                btn.click(handle_single_transcription, [f_in, l_in, t_in, m_in, live_in], [t_out, s_out, l_out, d_out, f_out], show_progress=False)

            with gr.TabItem("📦 Batch Transcribe"):
                with gr.Row():
                    with gr.Column(scale=1):
                        fb_in = gr.File(label="Files", file_count="multiple", file_types=all_media)
                        fl_in = gr.Dropdown(choices=l_choices, value="auto", label="Language")
                        fm_in = gr.Dropdown(choices=m_choices, value=default_model_val, label="Model")
                        ft_in = gr.Checkbox(label="Timestamps", value=False)
                        fbtn = gr.Button("⚡ Process Batch", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        ft_out = gr.Markdown("*Upload multiple files to begin...*")
                        fs_out = gr.Textbox(label="Status", interactive=False)
                        fb_files_out = gr.File(label="Download Individual Transcriptions", file_count="multiple")
                fbtn.click(handle_batch_transcription, [fb_in, fl_in, ft_in, fm_in], [ft_out, fs_out, fb_files_out], show_progress=False)

            with gr.TabItem("🗃️ Archive"):
                with gr.Row():
                    with gr.Column(scale=1):
                        arc_in = gr.File(label="Archive", file_types=[".zip", ".tar", ".gz", ".tgz"])
                        la_in = gr.Dropdown(choices=l_choices, value="auto", label="Language")
                        ma_in = gr.Dropdown(choices=m_choices, value=default_model_val, label="Model")
                        ta_in = gr.Checkbox(label="Timestamps", value=False)
                        btn_a = gr.Button("Extract & Transcribe", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        arc_out = gr.Markdown("*Upload archive to begin...*")
                        sas_out = gr.Textbox(label="Status", interactive=False)
                        farc_out = gr.File(label="Download Individual Transcriptions", file_count="multiple")
                btn_a.click(handle_archive_transcription, [arc_in, la_in, ta_in, ma_in], [arc_out, sas_out, farc_out])
    return interface

def main():
    interface = build_interface()
    # FIX: Added allowed_paths to allow Gradio to access and serve files from the /transcripts volume
    interface.launch(
        server_name="0.0.0.0", 
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "10002")),
        allowed_paths=[TRANSCRIPTS_VOLUME_DIR]
    )

if __name__ == "__main__":
    main()