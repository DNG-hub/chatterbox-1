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
    "cat_inner_voice": "Cat Inner Voice - Internal monologue, unspoken thoughts",
    "cat_serious": "Cat Serious - Authoritative, deepened, commanding tone",
    "voice_comm": "Voice Comm - Helmet radio channel with PTT click",
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
        "cat_inner_voice": _filter_cat_inner_voice,
        "cat_serious": _filter_cat_serious,
        "voice_comm": _filter_voice_comm,
    }

    if mode not in filters:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(filters.keys())}")

    return filters[mode](audio, sr)


# ============================================================
# REALISTIC STATIC GENERATOR
# ============================================================
#
# Based on real-world phenomena:
# - Atmospheric noise (lightning/sferics): Impulse-based with damped decay
# - Multipath fading: Rayleigh-distributed amplitude variations
# - Tape degradation: Hiss with dropouts and micro-variations
# - Vinyl artifacts: Random pops and crackle clusters
# - Radio interference: AM modulation and signal fading
#
# See STATIC_DESIGN.md for detailed documentation.
# ============================================================

# Mode-specific static configurations
STATIC_PRESETS = {
    "whisper_chorus": {
        # Intimate tape recorder aesthetic - aged, fragile recording
        "description": "Vintage tape hiss - intimate, captured moment",
        "base_type": "tape_hiss",
        "freq_low": 6000,
        "freq_high": 14000,
        "base_level": 0.02,
        # Fading - subtle drift like old equipment
        "fade_depth": 0.15,          # ±15% variation
        "fade_rate": 0.3,            # Slow drift
        "fade_complexity": 2,        # Simple variation
        # Impulses - rare soft micro-pops
        "impulse_density": 1.5,      # 1-2 per second
        "impulse_amplitude": 0.4,    # Soft
        "impulse_decay_ms": 8,       # Quick decay
        "impulse_clustering": 0.1,   # Not clustered
        # Dropouts - occasional tape wear
        "dropout_probability": 0.3,  # Per second
        "dropout_duration_ms": 30,
        "dropout_depth": 0.6,
    },
    "spore_cloud": {
        # Organic particle texture - alive, breathing, emergent
        "description": "Granular particles - organic, breathing presence",
        "base_type": "granular",
        "freq_low": 2000,
        "freq_high": 7000,
        "base_level": 0.025,
        # Fading - breathing pattern
        "fade_depth": 0.25,          # ±25% variation
        "fade_rate": 0.5,            # Breathing speed
        "fade_complexity": 3,        # Organic complexity
        # Impulses - particle clusters
        "impulse_density": 12,       # Many tiny particles
        "impulse_amplitude": 0.25,   # Very quiet individually
        "impulse_decay_ms": 3,       # Tiny grains
        "impulse_clustering": 0.7,   # Highly clustered
        # Dropouts - none (continuous texture)
        "dropout_probability": 0.0,
        "dropout_duration_ms": 0,
        "dropout_depth": 0.0,
    },
    "mycelium_pulse": {
        # Underground rumble - ancient, vast, patient
        "description": "Deep earth rumble - primordial network",
        "base_type": "rumble",
        "freq_low": 25,
        "freq_high": 180,
        "base_level": 0.03,
        # Fading - slow geological drift
        "fade_depth": 0.30,          # ±30% variation
        "fade_rate": 0.15,           # Very slow
        "fade_complexity": 2,        # Simple but deep
        # Impulses - deep thuds like earth settling
        "impulse_density": 0.7,      # Rare
        "impulse_amplitude": 1.2,    # Substantial
        "impulse_decay_ms": 45,      # Long rumbling decay
        "impulse_clustering": 0.3,   # Some clustering
        # Dropouts - pressure waves
        "dropout_probability": 0.2,
        "dropout_duration_ms": 150,  # Long, slow
        "dropout_depth": 0.4,        # Partial, like pressure change
    },
    "resonance_capture": {
        # Corrupted electronics - haunted, breaking down
        "description": "Degraded signal - corrupted, unstable reality",
        "base_type": "unstable",
        "freq_low": 100,
        "freq_high": 9000,
        "base_level": 0.04,
        # Fading - erratic Rayleigh-style
        "fade_depth": 0.40,          # ±40% - very unstable
        "fade_rate": 0.8,            # Erratic
        "fade_complexity": 5,        # Chaotic
        # Impulses - heavy glitches
        "impulse_density": 8,        # Frequent
        "impulse_amplitude": 0.9,    # Loud pops
        "impulse_decay_ms": 5,       # Sharp digital glitches
        "impulse_clustering": 0.5,   # Some bursts
        # Dropouts - signal breaking up
        "dropout_probability": 1.5,  # Frequent
        "dropout_duration_ms": 50,
        "dropout_depth": 0.85,       # Deep signal loss
        # Special: EVP-like sweep artifacts
        "sweep_probability": 0.4,    # Per second
    },
    "transmission": {
        # Radio broadcast - distant but intentional
        "description": "Radio static - broadcast breaking through",
        "base_type": "radio",
        "freq_low": 800,
        "freq_high": 5000,
        "base_level": 0.02,
        # Fading - multipath pattern
        "fade_depth": 0.20,          # ±20%
        "fade_rate": 0.4,            # Medium
        "fade_complexity": 4,        # Multipath complexity
        # Impulses - interference bursts
        "impulse_density": 3,        # Occasional
        "impulse_amplitude": 0.6,    # Moderate
        "impulse_decay_ms": 12,      # Radio crackle decay
        "impulse_clustering": 0.2,   # Mostly independent
        # Dropouts - atmospheric fades
        "dropout_probability": 0.5,
        "dropout_duration_ms": 80,
        "dropout_depth": 0.5,        # Partial fade
        # Special: AM wobble
        "am_wobble_depth": 0.15,
        "am_wobble_rate": 1.2,
    },
    "cat_inner_voice": {
        # Internal monologue - thoughts bypassing the ears entirely
        # Near-silent: the mind is a quiet place
        "description": "Neural whisper - thoughts forming before speech",
        "base_type": "tape_hiss",
        "freq_low": 4000,
        "freq_high": 10000,
        "base_level": 0.006,
        # Fading - very slow, meditative drift
        "fade_depth": 0.08,          # ±8% - barely perceptible
        "fade_rate": 0.2,            # Very slow, like breathing
        "fade_complexity": 1,        # Simple, calm
        # Impulses - rare faint neural flickers
        "impulse_density": 0.5,      # Very rare
        "impulse_amplitude": 0.15,   # Barely there
        "impulse_decay_ms": 4,       # Tiny sparks
        "impulse_clustering": 0.0,   # Independent
        # Dropouts - none (thoughts don't drop out)
        "dropout_probability": 0.0,
        "dropout_duration_ms": 0,
        "dropout_depth": 0.0,
    },
    "cat_serious": {
        # Minimal static - serious mode is clean and commanding
        "description": "Near silence - authority needs no noise",
        "base_type": "tape_hiss",
        "freq_low": 5000,
        "freq_high": 12000,
        "base_level": 0.004,
        # Fading - barely there
        "fade_depth": 0.05,
        "fade_rate": 0.15,
        "fade_complexity": 1,
        # Impulses - none
        "impulse_density": 0.0,
        "impulse_amplitude": 0.0,
        "impulse_decay_ms": 0,
        "impulse_clustering": 0.0,
        # Dropouts - none
        "dropout_probability": 0.0,
        "dropout_duration_ms": 0,
        "dropout_depth": 0.0,
    },
    "voice_comm": {
        # Radio channel static - constant background hiss with crackle
        "description": "Radio channel noise - helmet comms hiss",
        "base_type": "radio",
        "freq_low": 400,
        "freq_high": 3200,
        "base_level": 0.025,
        # Fading - slight signal variation
        "fade_depth": 0.12,
        "fade_rate": 0.5,
        "fade_complexity": 2,
        # Impulses - occasional radio crackle
        "impulse_density": 4,
        "impulse_amplitude": 0.3,
        "impulse_decay_ms": 6,
        "impulse_clustering": 0.3,
        # Dropouts - very rare brief signal dips
        "dropout_probability": 0.2,
        "dropout_duration_ms": 20,
        "dropout_depth": 0.3,
        # AM wobble - subtle
        "am_wobble_depth": 0.06,
        "am_wobble_rate": 0.8,
    },
}


def _generate_static(length: int, sr: int, mode: str, level: float = None) -> np.ndarray:
    """
    Generate realistic mode-specific static/noise layer.

    Implements real-world phenomena:
    - Amplitude fading (multipath/equipment drift)
    - Random impulse events (lightning/pops/glitches)
    - Signal dropouts (tape wear/signal loss)
    - Mode-specific characteristics

    Args:
        length: Number of samples
        sr: Sample rate
        mode: Ghost mode name
        level: Static level override (None = use preset default)

    Returns:
        Static noise array with realistic variations
    """
    if mode not in STATIC_PRESETS:
        mode = "resonance_capture"  # Fallback

    config = STATIC_PRESETS[mode]
    base_level = level if level is not None else config["base_level"]

    # Time array for modulation
    t = np.arange(length) / sr
    duration = length / sr

    # Helper for safe filter frequencies
    nyquist = sr / 2
    def safe_wn(freq):
        return max(0.01, min(0.99, freq / nyquist))

    # =========================================
    # 1. Generate base noise by type
    # =========================================
    noise = np.random.randn(length).astype(np.float32)

    if config["base_type"] == "tape_hiss":
        # High-frequency tape hiss with subtle texture
        b, a = signal.butter(4, safe_wn(config["freq_low"]), 'high')
        noise = signal.filtfilt(b, a, noise)
        b, a = signal.butter(4, safe_wn(config["freq_high"]), 'low')
        noise = signal.filtfilt(b, a, noise)

    elif config["base_type"] == "granular":
        # Mid-frequency with granular micro-texture
        b, a = signal.butter(3, [safe_wn(config["freq_low"]), safe_wn(config["freq_high"])], 'band')
        noise = signal.filtfilt(b, a, noise)
        # Add micro-granular texture
        grain_samples = int(sr * 0.003)  # 3ms grains
        grain_env = np.ones(length)
        for i in range(0, length, grain_samples):
            grain_env[i:min(i+grain_samples, length)] *= np.random.uniform(0.7, 1.3)
        noise = noise * grain_env

    elif config["base_type"] == "rumble":
        # Deep sub-bass rumble
        b, a = signal.butter(6, safe_wn(config["freq_high"]), 'low')
        noise = signal.filtfilt(b, a, noise)
        if config["freq_low"] > 20:
            b, a = signal.butter(2, safe_wn(config["freq_low"]), 'high')
            noise = signal.filtfilt(b, a, noise)

    elif config["base_type"] == "unstable":
        # Broadband with instability - frequency band drifts
        center = (config["freq_low"] + config["freq_high"]) / 2
        width = (config["freq_high"] - config["freq_low"]) / 2
        # Random drift in filter
        drift = np.random.uniform(-0.1, 0.1)
        low_f = max(50, config["freq_low"] * (1 + drift))
        high_f = min(sr/2 - 100, config["freq_high"] * (1 - drift))
        b, a = signal.butter(3, [safe_wn(low_f), safe_wn(high_f)], 'band')
        noise = signal.filtfilt(b, a, noise)

    elif config["base_type"] == "radio":
        # Radio band with characteristic texture
        b, a = signal.butter(4, [safe_wn(config["freq_low"]), safe_wn(config["freq_high"])], 'band')
        noise = signal.filtfilt(b, a, noise)
        # AM wobble characteristic of radio
        if "am_wobble_depth" in config:
            am_rate = config.get("am_wobble_rate", 1.0) + np.random.uniform(-0.3, 0.3)
            am_depth = config["am_wobble_depth"]
            am = 1.0 - am_depth + am_depth * (0.5 + 0.5 * np.sin(2 * np.pi * am_rate * t))
            noise = noise * am

    # Normalize base noise
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))

    # =========================================
    # 2. Apply amplitude fading envelope
    # =========================================
    fade_depth = config["fade_depth"]
    fade_rate = config["fade_rate"]
    fade_complexity = config["fade_complexity"]

    # Create complex fading by summing multiple sine waves at different rates
    fade_envelope = np.ones(length)
    for i in range(fade_complexity):
        rate = fade_rate * (0.5 + i * 0.4) + np.random.uniform(-0.1, 0.1)
        phase = np.random.uniform(0, 2 * np.pi)
        weight = 1.0 / (i + 1)  # Lower frequencies dominate
        fade_envelope += weight * fade_depth * np.sin(2 * np.pi * rate * t + phase)

    # Normalize envelope to center around 1.0
    fade_envelope = fade_envelope / fade_envelope.mean()
    # Clamp to reasonable range
    fade_envelope = np.clip(fade_envelope, 1.0 - fade_depth, 1.0 + fade_depth)

    noise = noise * fade_envelope

    # =========================================
    # 3. Add impulse events (pops, clicks, thuds)
    # =========================================
    impulse_density = config["impulse_density"]
    if impulse_density > 0:
        impulse_amp = config["impulse_amplitude"]
        impulse_decay_samples = int(config["impulse_decay_ms"] * sr / 1000)
        clustering = config["impulse_clustering"]

        # Number of impulses based on density and duration
        expected_impulses = int(impulse_density * duration)
        num_impulses = np.random.poisson(expected_impulses)

        # Generate impulse times with optional clustering
        if num_impulses > 0:
            impulse_times = []
            current_time = np.random.uniform(0, 0.5)  # Start within first 0.5s

            for _ in range(num_impulses):
                impulse_times.append(current_time)
                # Inter-arrival time: clustered = shorter gaps sometimes
                if np.random.random() < clustering:
                    gap = np.random.exponential(0.05)  # Clustered: short gap
                else:
                    gap = np.random.exponential(1.0 / impulse_density)  # Normal spacing
                current_time += gap
                if current_time >= duration:
                    break

            # Create impulse layer
            impulse_layer = np.zeros(length)
            for imp_time in impulse_times:
                imp_sample = int(imp_time * sr)
                if imp_sample < length:
                    # Impulse amplitude varies
                    amp = impulse_amp * np.random.uniform(0.5, 1.5)
                    # Damped decay envelope
                    decay_len = min(impulse_decay_samples, length - imp_sample)
                    decay = np.exp(-np.arange(decay_len) / (impulse_decay_samples / 3))
                    # Random polarity
                    polarity = np.random.choice([-1, 1])
                    impulse_layer[imp_sample:imp_sample + decay_len] += polarity * amp * decay * np.random.randn(decay_len)

            # Filter impulses to match mode frequency range
            if np.max(np.abs(impulse_layer)) > 0:
                try:
                    b, a = signal.butter(2, [safe_wn(config["freq_low"]), safe_wn(config["freq_high"])], 'band')
                    impulse_layer = signal.filtfilt(b, a, impulse_layer)
                except:
                    pass  # If filter fails, use unfiltered

            noise = noise + impulse_layer

    # =========================================
    # 4. Add dropouts (signal loss moments)
    # =========================================
    dropout_prob = config["dropout_probability"]
    if dropout_prob > 0:
        dropout_duration_samples = int(config["dropout_duration_ms"] * sr / 1000)
        dropout_depth = config["dropout_depth"]

        expected_dropouts = int(dropout_prob * duration)
        num_dropouts = np.random.poisson(expected_dropouts)

        dropout_envelope = np.ones(length)
        for _ in range(num_dropouts):
            # Random position
            drop_start = np.random.randint(0, max(1, length - dropout_duration_samples))
            drop_len = int(dropout_duration_samples * np.random.uniform(0.5, 1.5))
            drop_end = min(drop_start + drop_len, length)

            # Smooth fade in/out of dropout
            fade_samples = min(drop_len // 4, int(sr * 0.01))  # Max 10ms fade
            if fade_samples > 0 and drop_end - drop_start > fade_samples * 2:
                # Fade down
                dropout_envelope[drop_start:drop_start + fade_samples] *= np.linspace(1, 1 - dropout_depth, fade_samples)
                # Hold
                dropout_envelope[drop_start + fade_samples:drop_end - fade_samples] *= (1 - dropout_depth)
                # Fade up
                dropout_envelope[drop_end - fade_samples:drop_end] *= np.linspace(1 - dropout_depth, 1, fade_samples)

        noise = noise * dropout_envelope

    # =========================================
    # 5. Special effects (mode-specific)
    # =========================================

    # EVP-like sweep artifacts for resonance_capture
    if config.get("sweep_probability", 0) > 0:
        sweep_prob = config["sweep_probability"]
        expected_sweeps = int(sweep_prob * duration)
        num_sweeps = np.random.poisson(expected_sweeps)

        for _ in range(num_sweeps):
            sweep_start = np.random.randint(0, max(1, length - int(sr * 0.2)))
            sweep_len = int(sr * np.random.uniform(0.05, 0.15))
            sweep_end = min(sweep_start + sweep_len, length)

            # Create sweep: frequency-modulated noise burst
            sweep_t = np.arange(sweep_len) / sr
            sweep_freq = np.linspace(
                np.random.uniform(500, 2000),
                np.random.uniform(1500, 4000),
                sweep_len
            )
            sweep = 0.3 * np.sin(2 * np.pi * np.cumsum(sweep_freq) / sr)
            sweep *= np.hanning(sweep_len)  # Window it

            noise[sweep_start:sweep_end] += sweep[:sweep_end - sweep_start]

    # =========================================
    # 6. Final normalization and level application
    # =========================================
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))
    noise = noise * base_level

    return noise.astype(np.float32)


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


def _filter_cat_inner_voice(y: np.ndarray, sr: int) -> np.ndarray:
    """
    CAT INNER VOICE: Internal monologue - whispered unspoken thoughts

    Optimized for female voice (~165-255 Hz fundamental).
    Clean whisper effect via EQ sculpting + dynamics + subtle saturation.

    - Subtle saturation for closeness/density at low volume
    - Highpass removes chest/body resonance (whispers lack low-end power)
    - Lowpass softens brightness
    - Presence dip removes the "speaking out loud" quality
    - Sibilance dip (de-esser) at 5-8 kHz tames harsh S sounds
    - Warmth in the 800-1400 Hz zone for intimate female vocal character
    - Light compression for confessional steadiness (not squashed)
    - Subtle pitch shift down for "thought" quality
    - Tiny cranial reverb for inner-head intimacy
    """
    # --- Subtle pitch shift: thoughts sit slightly lower than speech ---
    y_whisper = librosa.effects.pitch_shift(y, sr=sr, n_steps=-0.3)
    y_whisper = 0.15 * y + 0.85 * y_whisper

    # --- Very subtle saturation: adds closeness and density at low volume ---
    y_whisper = np.tanh(y_whisper * 1.2) * 0.85

    # --- EQ shaping to remove "voiced" quality and create whisper ---
    nyquist = sr / 2

    # Presence dip at 3.2 kHz: removes "speaking out loud" quality
    if 3200 < nyquist:
        low = max(0.01, min(0.99, (3200 - 800) / nyquist))
        high = max(low + 0.01, min(0.99, (3200 + 800) / nyquist))
        b, a = signal.butter(2, [low, high], 'band')
        presence_band = signal.filtfilt(b, a, y_whisper)
        y_whisper = y_whisper - 0.40 * presence_band

    # Sibilance dip (de-esser): tame harsh S sounds at 5-8 kHz
    if 6000 < nyquist:
        low_s = max(0.01, min(0.99, 5000 / nyquist))
        high_s = max(low_s + 0.01, min(0.99, 8000 / nyquist))
        b, a = signal.butter(2, [low_s, high_s], 'band')
        sibilance_band = signal.filtfilt(b, a, y_whisper)
        y_whisper = y_whisper - 0.30 * sibilance_band

    # Warmth boost: intimate female vocal zone (800-1400 Hz)
    if 1000 < nyquist:
        low_w = max(0.01, min(0.99, 700 / nyquist))
        high_w = max(low_w + 0.01, min(0.99, 1400 / nyquist))
        b, a = signal.butter(2, [low_w, high_w], 'band')
        warmth_band = signal.filtfilt(b, a, y_whisper)
        y_whisper = y_whisper + 0.15 * warmth_band

    y_whisper = y_whisper / (np.max(np.abs(y_whisper)) + 1e-8) * 0.85

    # --- Pedalboard: shape into a clean whisper ---
    board = Pedalboard([
        # Highpass: remove chest resonance
        HighpassFilter(cutoff_frequency_hz=120),
        # Lowpass: soften brightness
        LowpassFilter(cutoff_frequency_hz=7500),
        # Light compression: confessional steadiness, not squashed
        Compressor(threshold_db=-24, ratio=3.5),
        # Tiny cranial reverb: intimate, inside-the-head
        Reverb(room_size=0.10, damping=0.88, wet_level=0.15, dry_level=0.85),
        # Reduce overall volume: whispers are quiet
        Gain(gain_db=-3),
        Limiter(threshold_db=-1),
    ])

    processed = board(y_whisper.reshape(1, -1), sr)
    processed = processed.flatten()

    # No static - clean whisper
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.75

    return processed


def _filter_cat_serious(y: np.ndarray, sr: int) -> np.ndarray:
    """
    CAT SERIOUS: Authoritative, deepened, commanding tone

    When Cat gets serious, her voice drops, becomes measured and controlled.
    Optimized for female voice shifting into commanding register.

    - Pitch drop (-3 semitones default, adjustable via parameterized version)
    - Low-mid boost for authority/gravitas (200-500 Hz)
    - Presence boost at 2-3 kHz for clarity and command
    - Tighter compression (controlled, deliberate delivery)
    - Reduced high frequencies (less playful brightness)
    - Minimal reverb (direct, no-nonsense)
    """
    # --- Pitch drop: the signature "getting serious" shift ---
    y_serious = librosa.effects.pitch_shift(y, sr=sr, n_steps=-3)

    # Blend: mostly pitched down, slight original for naturalness
    y_blended = 0.10 * y + 0.90 * y_serious
    y_blended = y_blended / (np.max(np.abs(y_blended)) + 1e-8) * 0.9

    # --- EQ shaping for authority ---
    nyquist = sr / 2

    # Low-mid boost: gravitas zone (200-500 Hz)
    if 500 < nyquist:
        low = max(0.01, min(0.99, 200 / nyquist))
        high = max(low + 0.01, min(0.99, 500 / nyquist))
        b, a = signal.butter(2, [low, high], 'band')
        gravitas_band = signal.filtfilt(b, a, y_blended)
        y_blended = y_blended + 0.20 * gravitas_band

    # Presence boost at 2-3 kHz: clarity and command
    if 2500 < nyquist:
        low_p = max(0.01, min(0.99, 2000 / nyquist))
        high_p = max(low_p + 0.01, min(0.99, 3000 / nyquist))
        b, a = signal.butter(2, [low_p, high_p], 'band')
        command_band = signal.filtfilt(b, a, y_blended)
        y_blended = y_blended + 0.12 * command_band

    y_blended = y_blended / (np.max(np.abs(y_blended)) + 1e-8) * 0.9

    # --- Pedalboard effects chain ---
    board = Pedalboard([
        # Highpass: clean low end, keep it tight
        HighpassFilter(cutoff_frequency_hz=80),
        # Lowpass: cut playful brightness
        LowpassFilter(cutoff_frequency_hz=8000),
        # Tight compression: controlled, deliberate
        Compressor(threshold_db=-20, ratio=4.5),
        # Minimal reverb: direct and commanding
        Reverb(room_size=0.08, damping=0.9, wet_level=0.08, dry_level=0.92),
        Gain(gain_db=1.5),
        Limiter(threshold_db=-0.5),
    ])

    processed = board(y_blended.reshape(1, -1), sr)
    processed = processed.flatten()

    # Very minimal static
    static = _generate_static(len(processed), sr, "cat_serious")
    processed = processed + static
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.9

    return processed


def _generate_ptt_click(sr: int, click_type: str = "release") -> np.ndarray:
    """
    Generate a pronounced, staticy push-to-talk click sound.

    Low-pitched, crunchy relay click with heavy static texture.
    Deep thump + mid snap + broadband static burst.
    Think military radio or aircraft comms squelch tail.

    Args:
        sr: Sample rate
        click_type: "press" or "release"

    Returns:
        Click audio array (~70-80ms)
    """
    if click_type == "release":
        duration_ms = 75
        click_amp = 0.85
    else:
        duration_ms = 55
        click_amp = 0.70

    num_samples = int(sr * duration_ms / 1000)
    t = np.arange(num_samples) / sr

    # Layer 1: Deep low thump (~200 Hz - chest-feel bass)
    thump = 0.8 * np.sin(2 * np.pi * 200 * t) * np.exp(-80 * t)

    # Layer 2: Mid snap (~500 Hz - relay contact)
    snap = 0.5 * np.sin(2 * np.pi * 500 * t) * np.exp(-150 * t)

    # Layer 3: Static burst - bandpassed noise throughout the click
    # This gives the click its staticy, crunchy radio character
    static_noise = np.random.randn(num_samples)
    # Bandpass the static to radio range (300-3000 Hz)
    nyq = sr / 2
    low_f = max(0.01, min(0.99, 300 / nyq))
    high_f = max(low_f + 0.01, min(0.99, 3000 / nyq))
    b, a = signal.butter(3, [low_f, high_f], 'band')
    static_noise = signal.filtfilt(b, a, static_noise)
    # Shape static with a decay envelope so it fades with the click
    static_env = np.exp(-40 * t)
    static_layer = 0.6 * static_noise * static_env

    # Layer 4: Sharp initial transient (mechanical impact)
    transient_len = min(int(sr * 0.005), num_samples)  # 5ms
    transient = np.zeros(num_samples)
    transient[:transient_len] = 0.7 * np.random.randn(transient_len)
    transient[:transient_len] *= np.exp(-np.arange(transient_len) / (transient_len / 2.5))

    click = thump + snap + static_layer + transient

    # Shape with a fast attack envelope
    attack_samples = min(int(sr * 0.001), num_samples)  # 1ms attack
    envelope = np.ones(num_samples)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    click = click * envelope

    # Normalize to target amplitude
    click = click / (np.max(np.abs(click)) + 1e-8) * click_amp

    return click.astype(np.float32)


def _filter_voice_comm(y: np.ndarray, sr: int) -> np.ndarray:
    """
    VOICE COMM: Helmet radio channel communication

    Simulates military/tactical voice comms through helmet-mounted radios.
    Unisex - the radio bandwidth flattens gender differences.

    - Tight bandpass (300 Hz - 3.4 kHz) - classic radio bandwidth
    - Nasal mid-boost (1-2 kHz) for radio bite
    - Mild saturation/clipping for radio limiter character
    - Heavy compression (radio AGC)
    - Static modulated INTO the voice signal (not background)
    - Noise gate / squelch: silence goes dead between speech
    - Slight boxy resonance (helmet cavity)
    - Pronounced low-pitch staticy PTT click at start and end of voice
    """
    # --- Mild saturation: radio limiter character ---
    y_radio = np.tanh(y * 1.8) * 0.7

    # --- Mix static INTO the voice signal ---
    static = _generate_static(len(y_radio), sr, "voice_comm")
    voice_envelope = np.abs(y_radio)
    voice_envelope = uniform_filter1d(voice_envelope, size=int(sr * 0.03))
    voice_envelope = voice_envelope / (np.max(voice_envelope) + 1e-8)
    modulated_static = static * (0.3 + 2.5 * voice_envelope)
    y_radio = y_radio + modulated_static

    # --- Nasal mid-boost at 1-2 kHz for radio bite ---
    nyquist = sr / 2
    if 1500 < nyquist:
        low_n = max(0.01, min(0.99, 1000 / nyquist))
        high_n = max(low_n + 0.01, min(0.99, 2000 / nyquist))
        b, a = signal.butter(2, [low_n, high_n], 'band')
        nasal_band = signal.filtfilt(b, a, y_radio)
        y_radio = y_radio + 0.18 * nasal_band

    y_radio = y_radio / (np.max(np.abs(y_radio)) + 1e-8) * 0.85

    # --- Pedalboard effects chain ---
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=300),
        LowpassFilter(cutoff_frequency_hz=3400),
        Compressor(threshold_db=-18, ratio=8),
        Reverb(room_size=0.05, damping=0.95, wet_level=0.06, dry_level=0.94),
        Gain(gain_db=2),
        Limiter(threshold_db=-1),
    ])

    processed = board(y_radio.reshape(1, -1), sr)
    processed = processed.flatten()
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.85

    # --- Noise gate / squelch: kill silence between speech ---
    # Detect voice activity with smoothed envelope
    gate_envelope = np.abs(processed)
    gate_envelope = uniform_filter1d(gate_envelope, size=int(sr * 0.02))
    gate_threshold = np.max(gate_envelope) * 0.04  # 4% of peak = gate open
    gate_mask = (gate_envelope > gate_threshold).astype(np.float32)
    # Smooth the gate edges (5ms attack, 30ms release) to avoid clicks
    gate_smooth = uniform_filter1d(gate_mask, size=int(sr * 0.03))
    gate_smooth = np.clip(gate_smooth * 3, 0, 1)  # Sharpen but keep smooth
    processed = processed * gate_smooth

    # --- Trim leading silence so press click is right before voice ---
    abs_signal = np.abs(processed)
    trim_threshold = np.max(abs_signal) * 0.02
    first_voice = 0
    while first_voice < len(processed) and abs_signal[first_voice] < trim_threshold:
        first_voice += 1
    # Keep a tiny 5ms lead-in before first voice for natural attack
    lead_samples = min(int(sr * 0.005), first_voice)
    processed = processed[max(0, first_voice - lead_samples):]

    # --- Trim trailing silence so release click is right at end of voice ---
    abs_signal = np.abs(processed)
    trim_threshold = np.max(abs_signal) * 0.02
    last_voice = len(processed) - 1
    while last_voice > 0 and abs_signal[last_voice] < trim_threshold:
        last_voice -= 1
    # Keep a tiny 5ms tail after last voice for natural decay
    tail_samples = min(int(sr * 0.005), len(processed) - last_voice - 1)
    processed = processed[:last_voice + tail_samples + 1]

    # --- PTT press click at start, release click at end ---
    press_click = _generate_ptt_click(sr, click_type="press")
    release_click = _generate_ptt_click(sr, click_type="release")
    # 30ms silence pad after release click so the ear registers it
    # before the file ends (without this, the decaying transient is
    # perceived as part of the voice tail-off, not a distinct click)
    release_pad = np.zeros(int(sr * 0.030))
    processed = np.concatenate([press_click, processed, release_click, release_pad])

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
