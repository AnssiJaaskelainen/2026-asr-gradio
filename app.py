"""
Simplest possible operational Gradio ASR app using faster-whisper.
Gradio 6.26.0 + faster-whisper 1.2.1
CPU and GPU compatible
"""
import os
import gradio as gr
from faster_whisper import WhisperModel

# Model configuration
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "medium")
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Global model instance
model = None

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

def transcribe(audio_path, language="auto", return_timestamps=False):
    """Transcribe audio file to text."""
    if not audio_path:
        return "", "No audio file provided", ""
    
    try:
        asr_model = load_model()
        beam_size = 5 if return_timestamps else 1
        word_timestamps = True if return_timestamps else False
        
        print(f"Transcribing: {audio_path}")
        segments, info = asr_model.transcribe(
            audio_path,
            language=language if language != "auto" else None,
            beam_size=beam_size,
            word_timestamps=word_timestamps
        )
        
        full_text = ""
        timestamp_lines = []
        
        for segment in segments:
            full_text += segment.text + " "
            if return_timestamps:
                timestamp_lines.append(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text.strip()}")
        
        lang_info = f"Detected: {info.language} ({info.language_probability:.1%})"
        model_info = f"Model: {MODEL_SIZE} | Device: {DEVICE}"
        timestamps_text = "\n".join(timestamp_lines) if timestamp_lines else ""
        
        return full_text.strip(), f"{lang_info}\n{model_info}", timestamps_text
        
    except Exception as e:
        return "", f"Error: {str(e)}", ""

# Build Gradio interface
with gr.Blocks(title="Whisper ASR") as demo:
    gr.Markdown("# 🎙️ Whisper ASR Transcription")
    gr.Markdown(f"**Model:** {MODEL_SIZE} | **Device:** {DEVICE}")
    
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
            transcribe_btn = gr.Button("🎯 Transcribe", variant="primary", size="lg")
        
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
    
    # Connect events
    transcribe_btn.click(
        fn=transcribe,
        inputs=[audio_input, language_dropdown, timestamps_check],
        outputs=[transcription_output, stats_output, timestamps_output]
    )

if __name__ == "__main__":
    print("=" * 50)
    print("Starting Whisper ASR Gradio App")
    print(f"Model: {MODEL_SIZE}")
    print(f"Device: {DEVICE}")
    print("=" * 50)
    demo.launch(
        server_name="0.0.0.0",
        server_port=10002,
        share=False
    )