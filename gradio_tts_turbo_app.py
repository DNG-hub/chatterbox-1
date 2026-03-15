import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts_turbo import ChatterboxTurboTTS

VOICES_DIR = r"E:\REPOS\StoryTeller\docs\voices"


def browse_voices_dialog():
    """Open a native OS file dialog starting at the voices directory."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    path = filedialog.askopenfilename(
        initialdir=VOICES_DIR,
        title="Select Reference Audio",
        filetypes=[
            ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path if path else None

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
        from ghost_filters_parameterized import apply_ghost_filter_parameterized
        import soundfile as sf
        import tempfile

        GHOST_FILTERS_AVAILABLE = True
        print("SUCCESS: Ghost filters loaded (with parameterized support)")

        def get_mode_names():
            return list(GHOST_MODES.keys())

        def get_mode_descriptions():
            return GHOST_MODES.copy()

        def apply_ghost_filter_to_gradio_audio(audio_tuple, mode):
            """Convert Gradio audio format, apply filter, return Gradio format (simple mode)"""
            if audio_tuple is None:
                return None
            
            # If mode is "None", return the original unfiltered audio
            if mode == "None":
                return audio_tuple

            sr, audio_data = audio_tuple

            # Ensure float32 format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                if np.abs(audio_data).max() > 1.0:
                    audio_data = audio_data / 32768.0

            # Apply filter
            filtered = apply_ghost_filter(audio_data, sr, mode)

            return (sr, filtered)

        def apply_ghost_filter_parameterized_to_gradio(
            audio_tuple, mode,
            # Global
            intensity, static_level_override,
            # Reverb
            reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
            # Chorus
            chorus_rate, chorus_depth, chorus_mix,
            # Delay
            delay_time, delay_feedback, delay_mix,
            # EQ & Dynamics
            highpass_cutoff, lowpass_cutoff,
            compressor_threshold, compressor_ratio, gain_db,
            # Whisper Chorus specific
            layer_mix_original, layer_mix_low, layer_mix_high, layer_mix_undertone,
            # Spore Cloud specific
            grain_rate, pulse_frequency, time_stretch_rate,
            # Mycelium Pulse specific
            pulse_rate, drone_frequency, drone_amplitude, pitch_shift_mycelium,
            # Resonance Capture specific
            bitcrush_depth, echo_mix,
            # Transmission specific
            pitch_shift_transmission,
            # Cat Inner Voice specific
            thought_pitch_shift, presence_dip, warmth_boost,
            cranial_reverb_size, thought_echo_mix, breathiness,
            # Cat Serious specific
            seriousness, gravitas_boost, command_presence,
            # Voice Comm specific
            radio_bandpass_low, radio_bandpass_high,
            saturation, helmet_resonance, click_volume,
        ):
            """Apply parameterized Ghost filter with full control"""
            if audio_tuple is None:
                return None

            # If mode is "None", return the original unfiltered audio
            if mode == "None":
                return audio_tuple

            sr, audio_data = audio_tuple

            # Ensure float32 format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                if np.abs(audio_data).max() > 1.0:
                    audio_data = audio_data / 32768.0

            # Convert "use default" sentinel values (negative means use default)
            def val_or_none(v, sentinel=-1):
                return None if v == sentinel else v

            # Static level: -1 means use mode default
            static_override = None if static_level_override < 0 else static_level_override

            # Apply parameterized filter
            filtered = apply_ghost_filter_parameterized(
                audio_data, sr, mode,
                # Global
                intensity=intensity,
                static_level_override=static_override,
                # Reverb
                reverb_room_size=reverb_room_size,
                reverb_damping=reverb_damping,
                reverb_wet_level=reverb_wet_level,
                reverb_dry_level=reverb_dry_level,
                # Chorus
                chorus_rate=chorus_rate,
                chorus_depth=chorus_depth,
                chorus_mix=chorus_mix,
                # Delay
                delay_time=delay_time,
                delay_feedback=delay_feedback,
                delay_mix=delay_mix,
                # EQ
                highpass_cutoff=highpass_cutoff,
                lowpass_cutoff=lowpass_cutoff,
                # Dynamics
                compressor_threshold=compressor_threshold,
                compressor_ratio=compressor_ratio,
                gain_db=gain_db,
                # Mode-specific (will be ignored if not applicable to mode)
                layer_mix_original=layer_mix_original,
                layer_mix_low=layer_mix_low,
                layer_mix_high=layer_mix_high,
                layer_mix_undertone=layer_mix_undertone,
                grain_rate=grain_rate,
                pulse_frequency=pulse_frequency,
                time_stretch_rate=time_stretch_rate,
                pulse_rate=pulse_rate,
                drone_frequency=drone_frequency,
                drone_amplitude=drone_amplitude,
                pitch_shift=pitch_shift_mycelium if mode == "mycelium_pulse" else pitch_shift_transmission,
                bitcrush_depth=int(bitcrush_depth),
                echo_mix=echo_mix,
                # Cat Inner Voice
                thought_pitch_shift=thought_pitch_shift,
                presence_dip=presence_dip,
                warmth_boost=warmth_boost,
                cranial_reverb_size=cranial_reverb_size,
                thought_echo_mix=thought_echo_mix,
                breathiness=breathiness,
                # Cat Serious
                seriousness=seriousness,
                gravitas_boost=gravitas_boost,
                command_presence=command_presence,
                # Voice Comm
                radio_bandpass_low=radio_bandpass_low,
                radio_bandpass_high=radio_bandpass_high,
                saturation=saturation,
                helmet_resonance=helmet_resonance,
                click_volume=click_volume,
            )

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
        browsed_audio_path,
        temperature,
        seed_num,
        min_p,
        top_p,
        top_k,
        repetition_penalty,
        norm_loudness,
        target_lufs
):
    if model is None:
        model = ChatterboxTurboTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    # Browsed path takes priority over mic recording
    effective_path = browsed_audio_path.strip() if browsed_audio_path and browsed_audio_path.strip() else audio_prompt_path

    wav = model.generate(
        text,
        audio_prompt_path=effective_path,
        temperature=temperature,
        min_p=min_p,
        top_p=top_p,
        top_k=int(top_k),
        repetition_penalty=repetition_penalty,
        norm_loudness=norm_loudness,
        target_lufs=target_lufs,
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
                    gr.Markdown("*Transform voice into Ghost's various communication modes*")

                    ghost_mode = gr.Dropdown(
                        choices=["None"] + get_mode_names(),
                        value="None",
                        label="Filter Mode",
                        info="Each mode has a unique sonic character"
                    )

                    # Mode description display
                    mode_descriptions = {
                        "None": "No filter applied",
                        "whisper_chorus": "Multiple consciousnesses speaking as one - intimate, layered voices",
                        "spore_cloud": "Voice condensing from particles - ethereal, granular texture",
                        "mycelium_pulse": "From the underground network - deep, rhythmic, subsonic",
                        "resonance_capture": "Corrupted recordings - glitchy, absorbed voices",
                        "transmission": "Direct broadcast - clean but otherworldly"
                    }
                    ghost_mode_info = gr.Markdown("*Select a mode to see description*")

                    # === Global Controls ===
                    with gr.Accordion("Global Controls", open=True):
                        gr.Markdown("**Master controls affecting the overall filter strength**")
                        ghost_intensity = gr.Slider(
                            minimum=0.0, maximum=2.0, value=1.0, step=0.05,
                            label="Filter Intensity",
                            info="0=None, 0.5=Subtle, 1.0=Default, 1.5=Strong, 2.0=Extreme"
                        )
                        ghost_static_level = gr.Slider(
                            minimum=-0.01, maximum=0.15, value=-0.01, step=0.005,
                            label="Static Level Override",
                            info="Background noise level. -0.01=Use mode default, 0=Silent, 0.02=Low, 0.05=Medium, 0.1=Heavy"
                        )

                    # === Reverb Section ===
                    with gr.Accordion("Reverb (Spatial/Echo)", open=False):
                        gr.Markdown("**Control the spatial characteristics and echo**")
                        with gr.Row():
                            ghost_reverb_room = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.4, step=0.05,
                                label="Room Size",
                                info="Larger = more cavernous. 0.2=Small, 0.5=Medium, 0.8=Cathedral"
                            )
                            ghost_reverb_damping = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                                label="Damping",
                                info="Higher = tighter reverb tail. 0.3=Echoey, 0.7=Natural, 0.9=Tight"
                            )
                        with gr.Row():
                            ghost_reverb_wet = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.25, step=0.05,
                                label="Wet Level",
                                info="Amount of reverb effect. 0.1=Subtle, 0.3=Noticeable, 0.6=Heavy"
                            )
                            ghost_reverb_dry = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.75, step=0.05,
                                label="Dry Level",
                                info="Amount of original signal. 0.4=Distant, 0.7=Balanced, 1.0=Present"
                            )

                    # === Chorus Section ===
                    with gr.Accordion("Chorus (Modulation/Doubling)", open=False):
                        gr.Markdown("**Add movement and thickness through modulation**")
                        with gr.Row():
                            ghost_chorus_rate = gr.Slider(
                                minimum=0.0, maximum=2.0, value=0.3, step=0.05,
                                label="Rate (Hz)",
                                info="Modulation speed. 0.1-0.3=Slow/subtle, 0.5-1.0=Noticeable, 1.5+=Fast warble"
                            )
                            ghost_chorus_depth = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.15, step=0.05,
                                label="Depth",
                                info="Pitch variation amount. 0.1=Subtle, 0.3=Moderate, 0.5+=Dramatic"
                            )
                        ghost_chorus_mix = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                            label="Mix",
                            info="Blend of chorus effect. 0.1=Hint, 0.3=Clear, 0.5+=Heavy"
                        )

                    # === Delay Section ===
                    with gr.Accordion("Delay (Echo/Repeat)", open=False):
                        gr.Markdown("**Add distinct echoes and repeats**")
                        with gr.Row():
                            ghost_delay_time = gr.Slider(
                                minimum=0.0, maximum=0.5, value=0.15, step=0.01,
                                label="Time (seconds)",
                                info="Echo delay. 0.05-0.1=Doubling, 0.15-0.25=Slap, 0.3-0.5=Distinct echo"
                            )
                            ghost_delay_feedback = gr.Slider(
                                minimum=0.0, maximum=0.9, value=0.2, step=0.05,
                                label="Feedback",
                                info="Echo repeats. 0=Single echo, 0.3=Few repeats, 0.6+=Cascading"
                            )
                        ghost_delay_mix = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.15, step=0.05,
                            label="Mix",
                            info="Echo volume. 0.1=Subtle, 0.25=Present, 0.5+=Prominent"
                        )

                    # === EQ & Dynamics Section ===
                    with gr.Accordion("EQ & Dynamics", open=False):
                        gr.Markdown("**Shape the frequency content and dynamics**")
                        with gr.Row():
                            ghost_highpass = gr.Slider(
                                minimum=20, maximum=2000, value=120, step=10,
                                label="Highpass Cutoff (Hz)",
                                info="Remove bass below this. 60=Full, 150=Thin, 500+=Telephone"
                            )
                            ghost_lowpass = gr.Slider(
                                minimum=1000, maximum=20000, value=12000, step=100,
                                label="Lowpass Cutoff (Hz)",
                                info="Remove treble above this. 4000=Muffled, 8000=Warm, 15000=Bright"
                            )
                        with gr.Row():
                            ghost_comp_threshold = gr.Slider(
                                minimum=-40, maximum=0, value=-20, step=1,
                                label="Compressor Threshold (dB)",
                                info="When compression starts. -30=Heavy, -20=Moderate, -10=Light"
                            )
                            ghost_comp_ratio = gr.Slider(
                                minimum=1, maximum=10, value=3, step=0.5,
                                label="Compressor Ratio",
                                info="Compression strength. 2=Gentle, 4=Moderate, 8+=Squashed"
                            )
                        ghost_gain = gr.Slider(
                            minimum=-10, maximum=10, value=0, step=0.5,
                            label="Output Gain (dB)",
                            info="Volume adjustment. -5=Quieter, 0=Unity, +5=Louder"
                        )

                    # === Mode-Specific: Whisper Chorus ===
                    with gr.Accordion("Whisper Chorus Settings", open=False, visible=False) as whisper_accordion:
                        gr.Markdown("**Layer mix for multiple voice effect**")
                        with gr.Row():
                            ghost_layer_original = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.55, step=0.05,
                                label="Original Voice",
                                info="Main voice layer volume"
                            )
                            ghost_layer_low = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.20, step=0.05,
                                label="Low Pitch Layer (-1 semi)",
                                info="Deeper undertone voice"
                            )
                        with gr.Row():
                            ghost_layer_high = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.15, step=0.05,
                                label="High Pitch Layer (+0.5 semi)",
                                info="Brighter overtone voice"
                            )
                            ghost_layer_undertone = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.10, step=0.05,
                                label="Undertone Layer (-3 semi)",
                                info="Deep whisper undertone - increase for 'multiple minds'"
                            )

                    # === Mode-Specific: Spore Cloud ===
                    with gr.Accordion("Spore Cloud Settings", open=False, visible=False) as spore_accordion:
                        gr.Markdown("**Granular/particle effect controls**")
                        with gr.Row():
                            ghost_grain_rate = gr.Slider(
                                minimum=1, maximum=20, value=8, step=1,
                                label="Grain Rate (Hz)",
                                info="Particle speed. 3-6=Organic/slow, 8-12=Default, 15+=Chaotic"
                            )
                            ghost_pulse_freq = gr.Slider(
                                minimum=0.1, maximum=2.0, value=0.5, step=0.1,
                                label="Pulse Frequency (Hz)",
                                info="Breathing rhythm. 0.3=Slow breath, 0.5=Natural, 1.0+=Fast"
                            )
                        ghost_time_stretch = gr.Slider(
                            minimum=0.95, maximum=1.1, value=1.02, step=0.01,
                            label="Time Stretch Rate",
                            info="Warping amount. 0.98=Subtle, 1.02=Default, 1.08=Heavy warp"
                        )

                    # === Mode-Specific: Mycelium Pulse ===
                    with gr.Accordion("Mycelium Pulse Settings", open=False, visible=False) as mycelium_accordion:
                        gr.Markdown("**Underground network effect controls**")
                        with gr.Row():
                            ghost_pulse_rate = gr.Slider(
                                minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                                label="Pulse Rate (Hz)",
                                info="Rhythmic pulse. 0.3=Slow/organic, 0.7=Default, 1.2+=Urgent"
                            )
                            ghost_pitch_mycelium = gr.Slider(
                                minimum=-5, maximum=5, value=-2, step=0.5,
                                label="Pitch Shift (semitones)",
                                info="Voice depth. -4=Very deep, -2=Underground, 0=Natural"
                            )
                        with gr.Row():
                            ghost_drone_freq = gr.Slider(
                                minimum=20, maximum=60, value=35, step=5,
                                label="Drone Frequency (Hz)",
                                info="Subsonic rumble pitch. 25=Very low, 35=Default, 50=Higher"
                            )
                            ghost_drone_amp = gr.Slider(
                                minimum=0.0, maximum=0.2, value=0.08, step=0.01,
                                label="Drone Amplitude",
                                info="Rumble volume. 0.04=Subtle, 0.08=Present, 0.15=Heavy"
                            )

                    # === Mode-Specific: Resonance Capture ===
                    with gr.Accordion("Resonance Capture Settings", open=False, visible=False) as resonance_accordion:
                        gr.Markdown("**Corruption/glitch effect controls**")
                        with gr.Row():
                            ghost_bitcrush = gr.Slider(
                                minimum=4, maximum=16, value=10, step=1,
                                label="Bitcrush Depth (bits)",
                                info="Digital degradation. 4-6=Heavy lo-fi, 10=Default, 14+=Subtle"
                            )
                            ghost_echo_mix = gr.Slider(
                                minimum=0.0, maximum=0.3, value=0.08, step=0.02,
                                label="Echo Mix",
                                info="Absorbed voice echo. 0.04=Hint, 0.08=Present, 0.2+=Haunted"
                            )

                    # === Mode-Specific: Transmission ===
                    with gr.Accordion("Transmission Settings", open=False, visible=False) as transmission_accordion:
                        gr.Markdown("**Broadcast quality controls**")
                        ghost_pitch_transmission = gr.Slider(
                            minimum=-5, maximum=5, value=-0.5, step=0.5,
                            label="Pitch Shift (semitones)",
                            info="Subtle otherworldly shift. -1 to +1 for broadcast, larger for character change"
                        )

                    # === Mode-Specific: Cat Inner Voice ===
                    with gr.Accordion("Cat Inner Voice Settings", open=False, visible=False) as cat_inner_accordion:
                        gr.Markdown("**Internal monologue controls - optimized for female voice**")
                        with gr.Row():
                            ghost_thought_pitch = gr.Slider(
                                minimum=-2.0, maximum=1.0, value=-0.3, step=0.1,
                                label="Thought Pitch Shift (semitones)",
                                info="Thought vs speech distinction. -0.5=Deeper thought, -0.3=Default, 0=Natural"
                            )
                            ghost_thought_echo = gr.Slider(
                                minimum=0.0, maximum=0.25, value=0.08, step=0.01,
                                label="Thought Echo Mix",
                                info="Faint inner echo. 0=None, 0.08=Default, 0.15=Echoey, 0.25=Dreamlike"
                            )
                        with gr.Row():
                            ghost_presence_dip = gr.Slider(
                                minimum=0.0, maximum=0.7, value=0.35, step=0.05,
                                label="Presence Dip",
                                info="Removes 'speaking aloud' quality. 0=None, 0.35=Default, 0.6=Very internal"
                            )
                            ghost_warmth_boost = gr.Slider(
                                minimum=0.0, maximum=0.4, value=0.15, step=0.05,
                                label="Warmth Boost",
                                info="Female vocal warmth (800-1400 Hz). 0=None, 0.15=Default, 0.3=Warm"
                            )
                        ghost_cranial_reverb = gr.Slider(
                            minimum=0.05, maximum=0.4, value=0.12, step=0.01,
                            label="Cranial Reverb Size",
                            info="Inner-head space. 0.05=Tight/dry, 0.12=Default, 0.3=Spacious mind"
                        )
                        ghost_breathiness = gr.Slider(
                            minimum=0.0, maximum=0.7, value=0.35, step=0.05,
                            label="Breathiness",
                            info="Whisper texture amount. 0=Clean, 0.2=Hint, 0.35=Default, 0.5+=Heavy whisper"
                        )

                    # === Mode-Specific: Cat Serious ===
                    with gr.Accordion("Cat Serious Settings", open=False, visible=False) as cat_serious_accordion:
                        gr.Markdown("**Seriousness grade controls - how commanding the tone becomes**")
                        ghost_seriousness = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                            label="Seriousness",
                            info="Grade of seriousness. 0.2=Slightly firm, 0.5=Default (-3 semi), 0.8=Very serious (-4.2 semi), 1.0=Maximum (-5 semi)"
                        )
                        with gr.Row():
                            ghost_gravitas = gr.Slider(
                                minimum=0.0, maximum=0.5, value=0.20, step=0.05,
                                label="Gravitas Boost",
                                info="Low-mid authority (200-500 Hz). 0=None, 0.2=Default, 0.4=Heavy"
                            )
                            ghost_command = gr.Slider(
                                minimum=0.0, maximum=0.3, value=0.12, step=0.02,
                                label="Command Presence",
                                info="Clarity/command boost (2-3 kHz). 0=Soft, 0.12=Default, 0.25=Sharp"
                            )

                    # === Mode-Specific: Voice Comm ===
                    with gr.Accordion("Voice Comm Settings", open=False, visible=False) as voice_comm_accordion:
                        gr.Markdown("**Helmet radio channel controls**")
                        with gr.Row():
                            ghost_radio_low = gr.Slider(
                                minimum=100, maximum=800, value=300, step=50,
                                label="Bandpass Low (Hz)",
                                info="Radio low cutoff. 200=Wider, 300=Standard, 500=Narrow/tinny"
                            )
                            ghost_radio_high = gr.Slider(
                                minimum=2000, maximum=6000, value=3400, step=100,
                                label="Bandpass High (Hz)",
                                info="Radio high cutoff. 2500=Very narrow, 3400=Standard, 5000=Wide/clear"
                            )
                        with gr.Row():
                            ghost_saturation = gr.Slider(
                                minimum=1.0, maximum=3.5, value=1.8, step=0.1,
                                label="Saturation",
                                info="Radio limiter drive. 1.0=Clean, 1.8=Default, 2.5=Gritty, 3.5=Heavy"
                            )
                            ghost_helmet = gr.Slider(
                                minimum=0.0, maximum=0.15, value=0.06, step=0.01,
                                label="Helmet Resonance",
                                info="Boxy helmet cavity reverb. 0=None, 0.06=Default, 0.12=Very boxy"
                            )
                        ghost_click_vol = gr.Slider(
                            minimum=0.0, maximum=0.8, value=0.45, step=0.05,
                            label="PTT Click Volume",
                            info="End-of-transmission click. 0=None, 0.3=Subtle, 0.45=Default, 0.7=Loud"
                        )

                    # Apply button and output
                    apply_ghost_filter_btn = gr.Button("Apply Ghost Filter", variant="secondary", size="lg")
                    filtered_audio_output = gr.Audio(label="Filtered Output")

            with gr.Accordion("Advanced Options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 2.0, step=.05, label="Temperature", value=0.8)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="Top P", value=0.95)
                top_k = gr.Slider(0, 1000, step=10, label="Top K", value=1000)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.05, label="Repetition Penalty", value=1.2)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="Min P (Set to 0 to disable)", value=0.00)
                norm_loudness = gr.Checkbox(value=True, label="Normalize Output Loudness")
                target_lufs = gr.Slider(-40, -10, step=1, label="Target LUFS (YouTube = -14, Dialogue Stem = -24)", value=-14)

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
            target_lufs,
        ],
        outputs=audio_output,
    )

    # === Ghost Filter Event Handlers ===
    if GHOST_FILTERS_AVAILABLE:
        # Mode change handler - update description and show/hide mode-specific accordions
        def on_mode_change(mode):
            descriptions = {
                "None": "*No filter applied - select a mode above*",
                "whisper_chorus": "**Whisper Chorus**: Multiple consciousnesses speaking as one. Intimate, layered voices with subtle delays creating the sense of many minds unified.",
                "spore_cloud": "**Spore Cloud**: Voice condensing from particles. Ethereal, granular texture with slow breathing pulses - like forming from mist.",
                "mycelium_pulse": "**Mycelium Pulse**: From the underground network. Deep, rhythmic, with subsonic drone - ancient and earthen.",
                "resonance_capture": "**Resonance Capture**: Corrupted recordings of absorbed voices. Glitchy dropouts, bitcrushed, haunted by echoes.",
                "transmission": "**Transmission**: Direct broadcast to the audience. Clean but subtly otherworldly - breaking the fourth wall.",
                "cat_inner_voice": "**Cat Inner Voice**: Internal monologue - thoughts heard directly inside the mind. Intimate, warm, closer than close. Optimized for female voice.",
                "cat_serious": "**Cat Serious**: Authoritative, deepened tone. Adjustable grades of seriousness - from slightly firm to commanding. Optimized for female voice.",
                "voice_comm": "**Voice Comm**: Helmet radio channel. Tight bandwidth, radio compression, channel noise, and PTT release click. Unisex."
            }
            desc = descriptions.get(mode, "*Select a mode*")

            # Visibility for mode-specific accordions
            whisper_vis = gr.update(visible=(mode == "whisper_chorus"))
            spore_vis = gr.update(visible=(mode == "spore_cloud"))
            mycelium_vis = gr.update(visible=(mode == "mycelium_pulse"))
            resonance_vis = gr.update(visible=(mode == "resonance_capture"))
            transmission_vis = gr.update(visible=(mode == "transmission"))
            cat_inner_vis = gr.update(visible=(mode == "cat_inner_voice"))
            cat_serious_vis = gr.update(visible=(mode == "cat_serious"))
            voice_comm_vis = gr.update(visible=(mode == "voice_comm"))

            return desc, whisper_vis, spore_vis, mycelium_vis, resonance_vis, transmission_vis, cat_inner_vis, cat_serious_vis, voice_comm_vis

        ghost_mode.change(
            fn=on_mode_change,
            inputs=[ghost_mode],
            outputs=[
                ghost_mode_info,
                whisper_accordion,
                spore_accordion,
                mycelium_accordion,
                resonance_accordion,
                transmission_accordion,
                cat_inner_accordion,
                cat_serious_accordion,
                voice_comm_accordion
            ]
        )

        # Apply filter with all parameters
        apply_ghost_filter_btn.click(
            fn=apply_ghost_filter_parameterized_to_gradio,
            inputs=[
                audio_output, ghost_mode,
                # Global
                ghost_intensity, ghost_static_level,
                # Reverb
                ghost_reverb_room, ghost_reverb_damping, ghost_reverb_wet, ghost_reverb_dry,
                # Chorus
                ghost_chorus_rate, ghost_chorus_depth, ghost_chorus_mix,
                # Delay
                ghost_delay_time, ghost_delay_feedback, ghost_delay_mix,
                # EQ & Dynamics
                ghost_highpass, ghost_lowpass,
                ghost_comp_threshold, ghost_comp_ratio, ghost_gain,
                # Whisper Chorus specific
                ghost_layer_original, ghost_layer_low, ghost_layer_high, ghost_layer_undertone,
                # Spore Cloud specific
                ghost_grain_rate, ghost_pulse_freq, ghost_time_stretch,
                # Mycelium Pulse specific
                ghost_pulse_rate, ghost_drone_freq, ghost_drone_amp, ghost_pitch_mycelium,
                # Resonance Capture specific
                ghost_bitcrush, ghost_echo_mix,
                # Transmission specific
                ghost_pitch_transmission,
                # Cat Inner Voice specific
                ghost_thought_pitch, ghost_presence_dip, ghost_warmth_boost,
                ghost_cranial_reverb, ghost_thought_echo, ghost_breathiness,
                # Cat Serious specific
                ghost_seriousness, ghost_gravitas, ghost_command,
                # Voice Comm specific
                ghost_radio_low, ghost_radio_high,
                ghost_saturation, ghost_helmet, ghost_click_vol,
            ],
            outputs=filtered_audio_output,
        )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)
