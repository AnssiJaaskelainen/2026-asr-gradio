"""
Gradio frontend for Whisper ASR.
Connects to FastAPI backend on port 10001.
Exposed on port 10002.
"""
import os
import gradio as gr
import requests

API_URL = os.environ.get("API_URL", "http://localhost:10001")

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "medium")
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


def transcribe(audio_path, language="auto", return_timestamps=False):
    """Transcribe audio file via FastAPI backend."""
    if not audio_path:
        return "", "No audio file provided", ""
    
    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f)}
            data = {
                "language": language,
                "return_timestamps": str(return_timestamps).lower()
            }
            
            response = requests.post(
                f"{API_URL}/transcribe",
                files=files,
                data=data,
                timeout=600
            )
        
        if response.status_code != 200:
            error_detail = response.json().get("detail", "Unknown error")
            return "", f"Error: {error_detail}", ""
        
        result = response.json()
        
        lang_info = f"Detected: {result['language']} ({result['language_probability']:.1%})"
        model_info = f"Model: {result['model']} | Device: {result['device']}"
        
        timestamp_lines = []
        if return_timestamps and result.get("segments"):
            for seg in result["segments"]:
                timestamp_lines.append(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
        
        return result["text"], f"{lang_info}\n{model_info}", "\n".join(timestamp_lines)
        
    except requests.exceptions.ConnectionError:
        return "", "Error: Cannot connect to API. Is the FastAPI server running on port 10001?", ""
    except Exception as e:
        return "", f"Error: {str(e)}", ""


with gr.Blocks(title="Whisper ASR") as demo:
    gr.Markdown("# Whisper ASR Transcription")
    gr.Markdown(f"**Model:** {MODEL_SIZE} | **Device:** {DEVICE} | **API:** {API_URL}")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Upload Audio/Video",
                type="filepath",
                sources=["upload"]
            )
            language_dropdown = gr.Dropdown(
                choices=["auto", "en", "fi", "sv", "de", "fr", "es", "zh", "ja", "ko", "ru", "pt", "it", "pl", "nl", "ar"],
                value="auto",
                label="Language"
            )
            timestamps_check = gr.Checkbox(
                label="Include Timestamps",
                value=False
            )
            transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            transcription_output = gr.Textbox(
                label="Transcription",
                lines=10
            )
            stats_output = gr.Textbox(
                label="Info",
                lines=2,
                interactive=False
            )
            timestamps_output = gr.Textbox(
                label="Timestamps",
                lines=5
            )
    
    transcribe_btn.click(
        fn=transcribe,
        inputs=[audio_input, language_dropdown, timestamps_check],
        outputs=[transcription_output, stats_output, timestamps_output]
    )

if __name__ == "__main__":
    print("=" * 50)
    print("Starting Whisper ASR Gradio Frontend")
    print(f"API URL: {API_URL}")
    print(f"Port: 10002")
    print("=" * 50)
    demo.launch(
        server_name="0.0.0.0",
        server_port=10002,
        share=False
    )