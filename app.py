"""
Gradio Web UI for Whisper ASR API.
"""
from __future__ import annotations

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import gradio as gr
import httpx

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("whisper-gradio")

API_BASE_URL = os.environ.get("API_URL", "http://api:10001")
API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "300"))

class WhisperAPIClient:
    def __init__(self, base_url: str = API_BASE_URL, timeout: int = API_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None
        self._health_info: dict = {}

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, follow_redirects=True)
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def health_check(self) -> bool:
        try:
            response = self.client.get("/health")
            response.raise_for_status()
            self._health_info = response.json()
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

    def get_default_model(self) -> str:
        # Read from API (which reads from models.json)
        return self._health_info.get("default_model", "")

    def get_device(self) -> str:
        device_info = self._health_info.get("device", {})
        return device_info if isinstance(device_info, str) else "unknown"

    def get_use_openvino(self) -> bool:
        return self._health_info.get("use_openvino", False)

    def transcribe(self, file_path: str, language: str = "auto", return_timestamps: bool = False, model: Optional[str] = None) -> dict:
        with open(file_path, "rb") as f:
            parts = [
                ("file", (Path(file_path).name, f)),
                ("language", (None, language)),
                ("return_timestamps", (None, str(return_timestamps).lower())),
            ]
            if model:
                parts.append(("model_id", (None, model)))

            logger.info(f"POST /transcribe | model_id: {model} | lang: {language}")
            response = self.client.post("/transcribe", files=parts)
            response.raise_for_status()
            return response.json()

    def transcribe_batch(self, file_paths: List[str], language: str = "auto", return_timestamps: bool = False, model: Optional[str] = None) -> dict:
        file_handles = [open(fp, "rb") for fp in file_paths]
        try:
            parts = [("files", (Path(fp).name, fh)) for fp, fh in zip(file_paths, file_handles)]
            parts.append(("language", (None, language)))
            parts.append(("return_timestamps", (None, str(return_timestamps).lower())))
            if model:
                parts.append(("model_id", (None, model)))

            logger.info(f"POST /transcribe-batch | files: {len(file_paths)} | model_id: {model}")
            response = self.client.post("/transcribe-batch", files=parts)
            response.raise_for_status()
            return response.json()
        finally:
            for fh in file_handles: fh.close()

    def transcribe_archive(self, archive_path: str, language: str = "auto", return_timestamps: bool = False, model: Optional[str] = None) -> dict:
        with open(archive_path, "rb") as f:
            parts = [
                ("archive", (Path(archive_path).name, f)),
                ("language", (None, language)),
                ("return_timestamps", (None, str(return_timestamps).lower())),
            ]
            if model:
                parts.append(("model_id", (None, model)))

            logger.info(f"POST /transcribe-archive | archive: {archive_path} | model_id: {model}")
            response = self.client.post("/transcribe-archive", files=parts)
            response.raise_for_status()
            return response.json()

api_client = WhisperAPIClient()

# =============================================================================
# UI Logic
# =============================================================================

def create_model_choices():
    models = api_client.get_models()
    # No hardcoded fallbacks here. If API is empty, dropdown is empty.
    return [(f"{m.get('name', m['id'])} ({m['id']})", m["id"]) for m in models]

def create_language_choices():
    langs = api_client.get_languages()
    return [(l["name"], l["code"]) for l in langs]

def format_single_result(result: dict) -> str:
    if "error" in result: return f"❌ Error: {result['error']}"
    text = result.get("text", "")
    lang = result.get("language", "unknown")
    prob = result.get("language_probability", 0.0)
    output = f"**Detected Language**: {lang} ({prob:.1%} confidence)\n\n{text}"
    if result.get("segments"):
        output += "\n\n**Timestamps:**\n" + "\n".join([f"[{s['start']}s - {s['end']}s] {s['text']}" for s in result["segments"][:50]])
    return output

def format_batch_results(results: List[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        name = r.get("filename", f"File {i}")
        if r.get("success"):
            lines.append(f"**{i}. {name}** ({r.get('language')}): {r.get('text', '')[:200]}...")
        else:
            lines.append(f"**{i}. {name}** ❌ Error: {r.get('error')}")
    return "\n\n".join(lines)

def transcribe_file(file, lang, timestamps, model):
    if not file: raise gr.Error("Please upload a file")
    result = api_client.transcribe(file.name, lang, timestamps, model)
    return format_single_result(result), f"✓ Done | Model: {result.get('model')}", f"{result.get('language')} ({result.get('language_probability', 0):.1%})"

def transcribe_batch(files, lang, timestamps, model):
    if not files: raise gr.Error("Please upload files")
    result = api_client.transcribe_batch([f.name for f in files], lang, timestamps, model)
    return format_batch_results(result.get("results", [])), f"✓ {result.get('successful')}/{result.get('total')} completed"

def transcribe_archive(archive, lang, timestamps, model):
    if not archive: raise gr.Error("Please upload an archive")
    result = api_client.transcribe_archive(archive.name, lang, timestamps, model)
    return format_batch_results(result.get("results", [])), f"✓ {result.get('successful')}/{result.get('total')} processed"

def save_transcription(text):
    if not text: raise gr.Error("No text to save")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text.replace("**", ""))
        return f.name

# =============================================================================
# Interface Builder
# =============================================================================

def build_interface():
    api_healthy = api_client.health_check()
    m_choices = create_model_choices()
    l_choices = create_language_choices()
    
    # Set default value from the first available model in the JSON list
    default_model_val = m_choices[0][1] if m_choices else None
    
    all_media = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"]

    with gr.Blocks() as interface:
        gr.Markdown("# 🎙️ Whisper ASR Interface")
        status = "🟢 API Connected" if api_healthy else "🔴 API Disconnected"
        gr.Markdown(status)

        with gr.Tabs():
            with gr.TabItem("📄 Single File"):
                with gr.Row():
                    with gr.Column():
                        f_in = gr.File(label="File", file_types=all_media)
                        l_in = gr.Dropdown(choices=l_choices, value="auto", label="Language")
                        m_in = gr.Dropdown(choices=m_choices, value=default_model_val, label="Model")
                        t_in = gr.Checkbox(label="Timestamps", value=False)
                        btn = gr.Button("Transcribe", variant="primary")
                    with gr.Column():
                        t_out = gr.Markdown("*Transcription...*")
                        s_out = gr.Textbox(label="Status", interactive=False)
                        l_out = gr.Textbox(label="Detected Language", interactive=False)
                        gr.Button("💾 Save").click(save_transcription, [t_out], [gr.File()])
                btn.click(transcribe_file, [f_in, l_in, t_in, m_in], [t_out, s_out, l_out])

            with gr.TabItem("📚 Batch Files"):
                with gr.Row():
                    with gr.Column():
                        fb_in = gr.File(label="Files", file_count="multiple", file_types=all_media)
                        lb_in = gr.Dropdown(choices=l_choices, value="auto", label="Language")
                        mb_in = gr.Dropdown(choices=m_choices, value=default_model_val, label="Model")
                        tb_in = gr.Checkbox(label="Timestamps", value=False)
                        btn_b = gr.Button("Transcribe All", variant="primary")
                    with gr.Column():
                        tb_out = gr.Textbox(label="Results", interactive=False, lines=20)
                        sb_out = gr.Textbox(label="Status", interactive=False)
                        gr.Button("💾 Save").click(save_transcription, [tb_out], [gr.File()])
                btn_b.click(transcribe_batch, [fb_in, lb_in, tb_in, mb_in], [tb_out, sb_out])

            with gr.TabItem("📦 Archive"):
                with gr.Row():
                    with gr.Column():
                        arc_in = gr.File(label="Archive", file_types=[".zip", ".tar", ".gz"])
                        la_in = gr.Dropdown(choices=l_choices, value="auto", label="Language")
                        ma_in = gr.Dropdown(choices=m_choices, value=default_model_val, label="Model")
                        ta_in = gr.Checkbox(label="Timestamps", value=False)
                        btn_a = gr.Button("Extract & Transcribe", variant="primary")
                    with gr.Column():
                        arc_out = gr.Textbox(label="Results", interactive=False, lines=20)
                        sas_out = gr.Textbox(label="Status", interactive=False)
                        gr.Button("💾 Save").click(save_transcription, [arc_out], [gr.File()])
                btn_a.click(transcribe_archive, [arc_in, la_in, ta_in, ma_in], [arc_out, sas_out])

    return interface

def main():
    interface = build_interface()
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "10002")))

if __name__ == "__main__":
    main()