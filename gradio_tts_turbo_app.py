import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts_turbo import ChatterboxTurboTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Voice Filters Integration ===
VOICE_FILTERS_ENABLED = True

# Try to import Ghost filters
GHOST_FILTERS_AVAILABLE = False
if VOICE_FILTERS_ENABLED:
    try:
        import sys
        import os
        # Add Voice Filters directory to path
        voice_filters_path = os.path.join(os.path.dirname(__file__), "src", "Voice Filters")
        if voice_filters_path not in sys.path:
            sys.path.insert(0, voice_filters_path)

        from ghost_filters import apply_ghost_filter, GHOST_MODES
        import soundfile as sf
        import tempfile

        GHOST_FILTERS_AVAILABLE = True
        print("SUCCESS: Ghost filters loaded")

        def get_mode_names():
            return list(GHOST_MODES.keys())

        def apply_ghost_filter_to_gradio_audio(audio_tuple, mode):
            """Convert Gradio audio format, apply filter, return Gradio format"""
            if audio_tuple is None or mode == "None":
                return None

            sr, audio_data = audio_tuple

            # Ensure float32 format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                if np.abs(audio_data).max() > 1.0:
                    audio_data = audio_data / 32768.0

            # Apply filter
            filtered = apply_ghost_filter(audio_data, sr, mode)

            return (sr, filtered)

    except ImportError as e:
        print(f"WARNING: Ghost filters not available: {e}")
    except Exception as e:
        print(f"WARNING: Ghost filters failed to load: {e}")

EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]

# --- REFINED CSS ---
# 1. tag-container: Forces the row to wrap items instead of scrolling. Removes borders/backgrounds.
# 2. tag-btn: Sets the specific look (indigo theme) and stops them from stretching.
CUSTOM_CSS = """
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important; /* This fixes the one-per-line issue */
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}

.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}

.tag-btn:hover {
    background: #c7d2fe !important;
    transform: translateY(-1px);
}
"""

INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#main_textbox textarea');
    if (!textarea) return current_text + " " + tag_val; 

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;

    let prefix = " ";
    let suffix = " ";

    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";

    if (end < current_text.length && current_text[end] === ' ') suffix = "";

    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    print(f"Loading Chatterbox-Turbo on {DEVICE}...")
    model = ChatterboxTurboTTS.from_pretrained(DEVICE)
    return model


def generate(
        model,
        text,
        audio_prompt_path,
        temperature,
        seed_num,
        min_p,
        top_p,
        top_k,
        repetition_penalty,
        norm_loudness
):
    if model is None:
        model = ChatterboxTurboTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        temperature=temperature,
        min_p=min_p,
        top_p=top_p,
        top_k=int(top_k),
        repetition_penalty=repetition_penalty,
        norm_loudness=norm_loudness,
    )
    return (model.sr, wav.squeeze(0).numpy())


with gr.Blocks(title="Chatterbox Turbo", css=CUSTOM_CSS) as demo:
    gr.Markdown("# ⚡ Chatterbox Turbo")

    model_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and um all that jazz. Would you like me to get some prices for you?",
                label="Text to synthesize (max chars 300)",
                max_lines=5,
                elem_id="main_textbox"
            )

            # --- Event Tags ---
            # Switched back to Row, but applied specific CSS to force wrapping
            with gr.Row(elem_classes=["tag-container"]):
                for tag in EVENT_TAGS:
                    # elem_classes targets the button specifically
                    btn = gr.Button(tag, elem_classes=["tag-btn"])

                    btn.click(
                        fn=None,
                        inputs=[btn, text],
                        outputs=text,
                        js=INSERT_TAG_JS
                    )

            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File",
                value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav"
            )

            run_btn = gr.Button("Generate ⚡", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

            # === Ghost Voice Filter UI ===
            if GHOST_FILTERS_AVAILABLE:
                with gr.Accordion("Ghost Voice Filter", open=False):
                    ghost_mode = gr.Dropdown(
                        choices=["None"] + get_mode_names(),
                        value="None",
                        label="Filter Mode"
                    )
                    apply_ghost_filter_btn = gr.Button("Apply Ghost Filter", variant="secondary")
                    filtered_audio_output = gr.Audio(label="Filtered Output")

            with gr.Accordion("Advanced Options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 2.0, step=.05, label="Temperature", value=0.8)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="Top P", value=0.95)
                top_k = gr.Slider(0, 1000, step=10, label="Top K", value=1000)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.05, label="Repetition Penalty", value=1.2)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="Min P (Set to 0 to disable)", value=0.00)
                norm_loudness = gr.Checkbox(value=True, label="Normalize Loudness (-27 LUFS)")

    demo.load(fn=load_model, inputs=[], outputs=model_state)

    run_btn.click(
        fn=generate,
        inputs=[
            model_state,
            text,
            ref_wav,
            temp,
            seed_num,
            min_p,
            top_p,
            top_k,
            repetition_penalty,
            norm_loudness,
        ],
        outputs=audio_output,
    )

    # === Ghost Filter Button Handler ===
    if GHOST_FILTERS_AVAILABLE:
        def simple_apply_filter(audio_tuple, mode):
            if audio_tuple is None or mode == "None":
                return None
            return apply_ghost_filter_to_gradio_audio(audio_tuple, mode)

        apply_ghost_filter_btn.click(
            fn=simple_apply_filter,
            inputs=[audio_output, ghost_mode],
            outputs=filtered_audio_output,
        )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)
