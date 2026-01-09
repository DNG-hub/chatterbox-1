"""
Ghost Voice Filters for Chatterbox Integration
Cat & Daniel: Collapse Protocol

Filters to transform TTS output into Ghost's various communication modes.

Usage in Gradio:
    from ghost_filters import apply_ghost_filter, GHOST_MODES

    # Get mode names for dropdown
    mode_names = list(GHOST_MODES.keys())

    # Apply filter to audio
    processed_audio = apply_ghost_filter(audio_array, sample_rate, mode="whisper_chorus")
"""

import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d
from scipy import signal
from pedalboard import (
    Pedalboard, Chorus, Reverb, Delay,
    HighpassFilter, LowpassFilter, Gain,
    Bitcrush, Compressor, Limiter
)

# ============================================================
# GHOST MODES - Available filter presets
# ============================================================

GHOST_MODES = {
    "whisper_chorus": "Whisper Chorus - Intimate, multiple minds speaking as one",
    "spore_cloud": "Spore Cloud - Ethereal, voice condensing from particles",
    "mycelium_pulse": "Mycelium Pulse - Deep, rhythmic, from underground network",
    "resonance_capture": "Resonance Capture - Corrupted, absorbed voices, glitched",
    "transmission": "Transmission - Direct audience address, broadcast quality",
}


def apply_ghost_filter(audio: np.ndarray, sr: int, mode: str = "whisper_chorus") -> np.ndarray:
    """
    Apply Ghost voice filter to audio.

    Args:
        audio: Audio array (mono, float)
        sr: Sample rate
        mode: One of GHOST_MODES keys

    Returns:
        Processed audio array
    """
    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Ensure float
    if audio.dtype != np.float32 and audio.dtype != np.float64:
        audio = audio.astype(np.float32)

    # Normalize input
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9

    # Apply mode-specific filter
    filters = {
        "whisper_chorus": _filter_whisper_chorus,
        "spore_cloud": _filter_spore_cloud,
        "mycelium_pulse": _filter_mycelium_pulse,
        "resonance_capture": _filter_resonance_capture,
        "transmission": _filter_transmission,
    }

    if mode not in filters:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(filters.keys())}")

    return filters[mode](audio, sr)


# ============================================================
# STATIC GENERATOR
# ============================================================

def _generate_static(length: int, sr: int, mode: str, level: float = None) -> np.ndarray:
    """
    Generate mode-specific static/noise layer.
    
    Args:
        length: Number of samples
        sr: Sample rate
        mode: Ghost mode name
        level: Static level (0.0-1.0), uses mode default if None
    
    Returns:
        Static noise array
    """
    # Mode-specific defaults
    mode_defaults = {
        "whisper_chorus": {
            "type": "tape_hiss",
            "level": 0.02,
            "highpass": 8000,
            "lowpass": 12000,
        },
        "spore_cloud": {
            "type": "crackle",
            "level": 0.025,
            "highpass": 2000,
            "lowpass": 5000,
        },
        "mycelium_pulse": {
            "type": "rumble",
            "level": 0.03,
            "highpass": 50,
            "lowpass": 200,
        },
        "resonance_capture": {
            "type": "white_noise",
            "level": 0.03,
            "highpass": 100,
            "lowpass": 8000,
        },
        "transmission": {
            "type": "radio_static",
            "level": 0.02,
            "highpass": 1000,
            "lowpass": 4000,
        },
    }
    
    if mode not in mode_defaults:
        mode = "resonance_capture"  # Fallback

    config = mode_defaults[mode]
    static_level = level if level is not None else config["level"]

    # Helper to clamp frequency to valid Nyquist range (0 < Wn < 1)
    nyquist = sr / 2
    def safe_wn(freq):
        wn = freq / nyquist
        return max(0.01, min(0.99, wn))  # Clamp to valid range

    # Generate base noise based on type
    if config["type"] == "tape_hiss":
        # High-frequency white noise (tape hiss)
        noise = np.random.randn(length).astype(np.float32)
        # Apply high-frequency emphasis
        b, a = signal.butter(4, safe_wn(config["highpass"]), 'high')
        noise = signal.filtfilt(b, a, noise)
        b, a = signal.butter(4, safe_wn(config["lowpass"]), 'low')
        noise = signal.filtfilt(b, a, noise)
        
    elif config["type"] == "crackle":
        # Mid-range crackle with granular texture
        noise = np.random.randn(length).astype(np.float32)
        # Add granular bursts
        grain_size = int(sr * 0.005)  # 5ms grains
        for i in range(0, length, grain_size):
            if np.random.random() < 0.3:  # 30% chance of grain
                end = min(i + grain_size, length)
                noise[i:end] *= np.random.uniform(1.5, 3.0)
        # Filter to mid-range
        b, a = signal.butter(4, [safe_wn(config["highpass"]), safe_wn(config["lowpass"])], 'band')
        noise = signal.filtfilt(b, a, noise)

    elif config["type"] == "rumble":
        # Low-frequency rumble
        noise = np.random.randn(length).astype(np.float32)
        # Heavy low-pass for rumble
        b, a = signal.butter(6, safe_wn(config["lowpass"]), 'low')
        noise = signal.filtfilt(b, a, noise)
        # Add some modulation for organic feel
        t = np.arange(length) / sr
        mod = 0.7 + 0.3 * np.sin(2 * np.pi * 0.3 * t)
        noise = noise * mod

    elif config["type"] == "white_noise":
        # Broad-spectrum white noise
        noise = np.random.randn(length).astype(np.float32)
        # Band-limit
        b, a = signal.butter(4, [safe_wn(config["highpass"]), safe_wn(config["lowpass"])], 'band')
        noise = signal.filtfilt(b, a, noise)

    elif config["type"] == "radio_static":
        # Radio interference static
        noise = np.random.randn(length).astype(np.float32)
        # Add some amplitude modulation for radio feel
        t = np.arange(length) / sr
        am_freq = np.random.uniform(0.5, 2.0)  # Random AM frequency
        am = 0.6 + 0.4 * np.sin(2 * np.pi * am_freq * t)
        noise = noise * am
        # Filter to radio band
        b, a = signal.butter(4, [safe_wn(config["highpass"]), safe_wn(config["lowpass"])], 'band')
        noise = signal.filtfilt(b, a, noise)
        
    else:
        # Fallback to white noise
        noise = np.random.randn(length).astype(np.float32)
    
    # Normalize and apply level
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))
    noise = noise * static_level
    
    return noise


# ============================================================
# FILTER IMPLEMENTATIONS
# ============================================================

def _filter_whisper_chorus(y: np.ndarray, sr: int) -> np.ndarray:
    """
    WHISPER CHORUS: Multiple consciousnesses speaking as one

    - 4 pitch-shifted layers with delays
    - Soft reverb for ethereal quality
    - Subtle chorus effect
    """
    # Layer 1: Original
    layer1 = y.copy()

    # Layer 2: Pitched down -1 semitone, delayed 30ms
    layer2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1)
    delay_samples = int(0.030 * sr)
    layer2 = np.pad(layer2, (delay_samples, 0))[:len(y)]

    # Layer 3: Pitched up +0.5 semitone, delayed 50ms
    layer3 = librosa.effects.pitch_shift(y, sr=sr, n_steps=0.5)
    delay_samples = int(0.050 * sr)
    layer3 = np.pad(layer3, (delay_samples, 0))[:len(y)]

    # Layer 4: Pitched down -3 semitones (whisper undertone), delayed 70ms
    layer4 = librosa.effects.pitch_shift(y, sr=sr, n_steps=-3)
    delay_samples = int(0.070 * sr)
    layer4 = np.pad(layer4, (delay_samples, 0))[:len(y)]

    # Mix layers
    mixed = (0.55 * layer1 + 0.20 * layer2 + 0.15 * layer3 + 0.10 * layer4)
    mixed = mixed / np.max(np.abs(mixed)) * 0.85

    # Pedalboard effects
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=120),
        Chorus(rate_hz=0.3, depth=0.15, mix=0.2),
        Reverb(room_size=0.4, damping=0.7, wet_level=0.25, dry_level=0.75),
        Compressor(threshold_db=-20, ratio=3),
        Limiter(threshold_db=-1),
    ])

    processed = board(mixed.reshape(1, -1), sr)
    processed = processed.flatten()
    
    # Add mode-specific static
    static = _generate_static(len(processed), sr, "whisper_chorus")
    processed = processed + static
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.9
    
    return processed


def _filter_spore_cloud(y: np.ndarray, sr: int) -> np.ndarray:
    """
    SPORE CLOUD: Voice condensing from particles

    - Granular-style amplitude modulation
    - Slow pulsing effect
    - Heavy reverb with long tail
    """
    t = np.arange(len(y)) / sr

    # Granular amplitude envelope
    grain_rate = 8  # Hz
    noise = np.random.randn(len(y))
    smoothed_noise = uniform_filter1d(noise, size=int(sr / grain_rate))
    grain_envelope = 0.65 + 0.35 * (smoothed_noise / np.max(np.abs(smoothed_noise)))

    # Slow pulse
    slow_pulse = 0.85 + 0.15 * np.sin(2 * np.pi * 0.5 * t)

    # Apply envelopes
    y_granular = y * grain_envelope * slow_pulse

    # Slight time stretch for warping
    y_stretched = librosa.effects.time_stretch(y_granular, rate=1.02)
    if len(y_stretched) > len(y):
        y_granular = y_stretched[:len(y)]
    else:
        y_granular = np.pad(y_stretched, (0, len(y) - len(y_stretched)))

    y_granular = y_granular / np.max(np.abs(y_granular)) * 0.8

    # Pedalboard effects
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=200),
        LowpassFilter(cutoff_frequency_hz=8000),
        Chorus(rate_hz=0.2, depth=0.3, mix=0.3),
        Reverb(room_size=0.7, damping=0.5, wet_level=0.45, dry_level=0.55),
        Delay(delay_seconds=0.15, feedback=0.2, mix=0.15),
        Compressor(threshold_db=-18, ratio=4),
        Limiter(threshold_db=-1),
    ])

    processed = board(y_granular.reshape(1, -1), sr)
    processed = processed.flatten()
    
    # Add mode-specific static
    static = _generate_static(len(processed), sr, "spore_cloud")
    processed = processed + static
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.8
    
    return processed


def _filter_mycelium_pulse(y: np.ndarray, sr: int) -> np.ndarray:
    """
    MYCELIUM PULSE: Voice from underground network

    - Low-frequency rhythmic modulation
    - Subsonic drone layer
    - Pitched down for depth
    """
    t = np.arange(len(y)) / sr

    # Organic pulsing rhythm
    pulse_rate = 0.7  # Hz
    pulse = 0.75 + 0.25 * np.sin(2 * np.pi * pulse_rate * t)
    slow_mod = 0.9 + 0.1 * np.sin(2 * np.pi * 0.15 * t)

    y_pulsed = y * pulse * slow_mod

    # Pitch down for underground feel
    y_deep = librosa.effects.pitch_shift(y_pulsed, sr=sr, n_steps=-2)

    # Subsonic drone
    drone_freq = 35  # Hz
    drone = 0.08 * np.sin(2 * np.pi * drone_freq * t)
    speech_envelope = np.abs(librosa.effects.preemphasis(y))
    speech_envelope = uniform_filter1d(speech_envelope, size=int(sr * 0.1))
    speech_envelope = speech_envelope / (np.max(speech_envelope) + 1e-8)
    drone = drone * speech_envelope

    y_mixed = y_deep + drone
    y_mixed = y_mixed / np.max(np.abs(y_mixed)) * 0.85

    # Pedalboard effects
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=60),
        LowpassFilter(cutoff_frequency_hz=6000),
        Reverb(room_size=0.6, damping=0.8, wet_level=0.35, dry_level=0.65),
        Compressor(threshold_db=-15, ratio=4),
        Gain(gain_db=2),
        Limiter(threshold_db=-1),
    ])

    processed = board(y_mixed.reshape(1, -1), sr)
    processed = processed.flatten()
    
    # Add mode-specific static
    static = _generate_static(len(processed), sr, "mycelium_pulse")
    processed = processed + static
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.85
    
    return processed


def _filter_resonance_capture(y: np.ndarray, sr: int) -> np.ndarray:
    """
    RESONANCE CAPTURE: Corrupted recordings, absorbed voices

    - Static bursts
    - Dropout glitches
    - Bitcrushing
    - Echo of absorbed voice
    """
    t = np.arange(len(y)) / sr

    # Dropout effect
    dropout_envelope = np.ones(len(y))
    num_dropouts = max(1, int(len(y) / sr * 0.5))
    for _ in range(num_dropouts):
        start = np.random.randint(0, max(1, len(y) - int(0.1 * sr)))
        length = int(np.random.uniform(0.02, 0.08) * sr)
        end = min(start + length, len(y))
        fade_len = max(1, length // 4)
        if start + fade_len < end - fade_len:
            dropout_envelope[start:start+fade_len] = np.linspace(1, 0.1, fade_len)
            dropout_envelope[start+fade_len:end-fade_len] = 0.1
            dropout_envelope[end-fade_len:end] = np.linspace(0.1, 1, fade_len)

    y_glitched = y * dropout_envelope

    # Echo of absorbed voice
    echo_voice = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
    delay_samples = int(0.1 * sr)
    echo_voice = np.pad(echo_voice, (delay_samples, 0))[:len(y)]

    y_mixed = y_glitched + 0.08 * echo_voice
    y_mixed = y_mixed / np.max(np.abs(y_mixed)) * 0.9

    # Pedalboard effects
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=200),
        Bitcrush(bit_depth=10),
        LowpassFilter(cutoff_frequency_hz=7000),
        Reverb(room_size=0.3, damping=0.9, wet_level=0.2, dry_level=0.8),
        Compressor(threshold_db=-18, ratio=5),
        Limiter(threshold_db=-1),
    ])

    processed = board(y_mixed.reshape(1, -1), sr)
    processed = processed.flatten()
    
    # Add mode-specific static
    static = _generate_static(len(processed), sr, "resonance_capture")
    processed = processed + static
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.9
    
    return processed


def _filter_transmission(y: np.ndarray, sr: int) -> np.ndarray:
    """
    TRANSMISSION: Direct audience address, broadcast quality

    - Clean but with subtle broadcast character
    - Minimal processing
    - Slight otherworldly pitch shift
    """
    # Subtle pitch shift
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=-0.5)
    y_blended = 0.7 * y + 0.3 * y_shifted
    y_blended = y_blended / np.max(np.abs(y_blended)) * 0.9

    # Pedalboard effects - cleaner than other modes
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=100),
        LowpassFilter(cutoff_frequency_hz=12000),
        Compressor(threshold_db=-18, ratio=3),
        Reverb(room_size=0.2, damping=0.8, wet_level=0.1, dry_level=0.9),
        Gain(gain_db=1),
        Limiter(threshold_db=-0.5),
    ])

    processed = board(y_blended.reshape(1, -1), sr)
    processed = processed.flatten()
    
    # Add mode-specific static
    static = _generate_static(len(processed), sr, "transmission")
    processed = processed + static
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.9
    
    return processed


# ============================================================
# FILE PROCESSING UTILITIES
# ============================================================

def process_file(input_path: str, output_path: str, mode: str = "whisper_chorus"):
    """
    Process a WAV file with Ghost filter.

    Args:
        input_path: Path to input WAV file
        output_path: Path to save processed WAV file
        mode: Ghost mode to apply
    """
    import soundfile as sf

    y, sr = librosa.load(input_path, sr=None)
    processed = apply_ghost_filter(y, sr, mode)
    sf.write(output_path, processed, sr)
    return output_path


def get_mode_names() -> list:
    """Get list of available mode names for UI dropdown."""
    return list(GHOST_MODES.keys())


def get_mode_descriptions() -> dict:
    """Get mode names with descriptions for UI."""
    return GHOST_MODES.copy()


# ============================================================
# GRADIO INTEGRATION HELPER
# ============================================================

def create_gradio_filter_component():
    """
    Returns Gradio components for Ghost filter integration.

    Usage in Chatterbox app.py:
        from ghost_filters import create_gradio_filter_component

        filter_dropdown, filter_button, filter_output = create_gradio_filter_component()
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio not installed. Run: pip install gradio")

    mode_choices = [(v, k) for k, v in GHOST_MODES.items()]

    dropdown = gr.Dropdown(
        choices=mode_choices,
        value="whisper_chorus",
        label="Ghost Voice Mode",
        info="Select filter to apply to generated audio"
    )

    button = gr.Button("Apply Ghost Filter", variant="secondary")

    return dropdown, button


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    print("Ghost Voice Filters")
    print("=" * 40)
    print("\nAvailable modes:")
    for mode, desc in GHOST_MODES.items():
        print(f"  - {mode}: {desc}")
    print("\nImport and use:")
    print("  from ghost_filters import apply_ghost_filter")
    print("  processed = apply_ghost_filter(audio, sr, mode='whisper_chorus')")
