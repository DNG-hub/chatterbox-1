"""
Parameterized Ghost Voice Filters - Enhanced version with adjustable parameters
Extends ghost_filters.py to support real-time parameter tweaking
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

# Import base filters - handle both relative and absolute imports
try:
    from .ghost_filters import (
        apply_ghost_filter as base_apply_ghost_filter,
        GHOST_MODES,
        STATIC_PRESETS,
        get_mode_names,
        _generate_static,
        _generate_ptt_click
    )
except ImportError:
    # Fallback for direct import
    from ghost_filters import (
        apply_ghost_filter as base_apply_ghost_filter,
        GHOST_MODES,
        STATIC_PRESETS,
        get_mode_names,
        _generate_static,
        _generate_ptt_click
    )


def apply_ghost_filter_parameterized(
    audio: np.ndarray,
    sr: int,
    mode: str = "whisper_chorus",
    # Reverb parameters
    reverb_room_size: float = None,
    reverb_damping: float = None,
    reverb_wet_level: float = None,
    reverb_dry_level: float = None,
    # Chorus parameters
    chorus_rate: float = None,
    chorus_depth: float = None,
    chorus_mix: float = None,
    # Delay parameters
    delay_time: float = None,
    delay_feedback: float = None,
    delay_mix: float = None,
    # EQ parameters
    highpass_cutoff: float = None,
    lowpass_cutoff: float = None,
    # Compression
    compressor_threshold: float = None,
    compressor_ratio: float = None,
    # Gain
    gain_db: float = None,
    # Mode-specific parameters
    intensity: float = 1.0,  # Overall filter intensity (0.0 to 2.0)
    # Whisper Chorus specific
    layer_mix_original: float = None,
    layer_mix_low: float = None,
    layer_mix_high: float = None,
    layer_mix_undertone: float = None,
    # Spore Cloud specific
    grain_rate: float = None,
    pulse_frequency: float = None,
    time_stretch_rate: float = None,
    # Mycelium Pulse specific
    pulse_rate: float = None,
    drone_frequency: float = None,
    drone_amplitude: float = None,
    pitch_shift: float = None,
    # Resonance Capture specific
    static_level: float = None,
    bitcrush_depth: int = None,
    echo_mix: float = None,
    # Cat Inner Voice specific
    thought_pitch_shift: float = None,
    presence_dip: float = None,
    warmth_boost: float = None,
    cranial_reverb_size: float = None,
    thought_echo_mix: float = None,
    breathiness: float = None,
    # Cat Serious specific
    seriousness: float = None,
    gravitas_boost: float = None,
    command_presence: float = None,
    # Voice Comm specific
    radio_bandpass_low: float = None,
    radio_bandpass_high: float = None,
    saturation: float = None,
    helmet_resonance: float = None,
    click_volume: float = None,
    # Static parameters (all modes)
    static_level_override: float = None,  # Override mode default static level
) -> np.ndarray:
    """
    Apply Ghost filter with adjustable parameters.
    Parameters set to None use mode defaults.
    """
    # Ensure mono and float
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32 and audio.dtype != np.float64:
        audio = audio.astype(np.float32)
    
    # Normalize input
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    
    # Get mode-specific defaults and apply parameterized filter
    if mode == "whisper_chorus":
        return _filter_whisper_chorus_param(
            audio, sr, intensity,
            reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
            chorus_rate, chorus_depth, chorus_mix,
            highpass_cutoff, compressor_threshold, compressor_ratio,
            layer_mix_original, layer_mix_low, layer_mix_high, layer_mix_undertone,
            static_level_override
        )
    elif mode == "spore_cloud":
        return _filter_spore_cloud_param(
            audio, sr, intensity,
            reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
            chorus_rate, chorus_depth, chorus_mix,
            delay_time, delay_feedback, delay_mix,
            highpass_cutoff, lowpass_cutoff, compressor_threshold, compressor_ratio,
            grain_rate, pulse_frequency, time_stretch_rate,
            static_level_override
        )
    elif mode == "mycelium_pulse":
        return _filter_mycelium_pulse_param(
            audio, sr, intensity,
            reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
            highpass_cutoff, lowpass_cutoff, compressor_threshold, compressor_ratio,
            gain_db, pulse_rate, drone_frequency, drone_amplitude, pitch_shift,
            static_level_override
        )
    elif mode == "resonance_capture":
        return _filter_resonance_capture_param(
            audio, sr, intensity,
            reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
            highpass_cutoff, lowpass_cutoff, compressor_threshold, compressor_ratio,
            static_level, bitcrush_depth, echo_mix,
            static_level_override
        )
    elif mode == "transmission":
        return _filter_transmission_param(
            audio, sr, intensity,
            reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
            highpass_cutoff, lowpass_cutoff, compressor_threshold, compressor_ratio,
            gain_db, pitch_shift,
            static_level_override
        )
    elif mode == "cat_inner_voice":
        return _filter_cat_inner_voice_param(
            audio, sr, intensity,
            reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
            chorus_rate, chorus_depth, chorus_mix,
            highpass_cutoff, lowpass_cutoff, compressor_threshold, compressor_ratio,
            thought_pitch_shift, presence_dip, warmth_boost,
            cranial_reverb_size, thought_echo_mix, breathiness,
            static_level_override
        )
    elif mode == "cat_serious":
        return _filter_cat_serious_param(
            audio, sr, intensity,
            reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
            highpass_cutoff, lowpass_cutoff, compressor_threshold, compressor_ratio,
            gain_db, seriousness, gravitas_boost, command_presence,
            static_level_override
        )
    elif mode == "voice_comm":
        return _filter_voice_comm_param(
            audio, sr, intensity,
            compressor_threshold, compressor_ratio, gain_db,
            radio_bandpass_low, radio_bandpass_high,
            saturation, helmet_resonance, click_volume,
            static_level_override
        )
    else:
        # Fallback to base implementation
        return base_apply_ghost_filter(audio, sr, mode)


def _filter_whisper_chorus_param(
    y: np.ndarray, sr: int, intensity: float,
    reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
    chorus_rate, chorus_depth, chorus_mix,
    highpass_cutoff, compressor_threshold, compressor_ratio,
    layer_mix_original, layer_mix_low, layer_mix_high, layer_mix_undertone,
    static_level_override
) -> np.ndarray:
    """Parameterized Whisper Chorus filter."""
    # Defaults
    defaults = {
        'layer_mix_original': 0.55,
        'layer_mix_low': 0.20,
        'layer_mix_high': 0.15,
        'layer_mix_undertone': 0.10,
        'reverb_room_size': 0.4,
        'reverb_damping': 0.7,
        'reverb_wet_level': 0.25,
        'reverb_dry_level': 0.75,
        'chorus_rate': 0.3,
        'chorus_depth': 0.15,
        'chorus_mix': 0.2,
        'highpass_cutoff': 120,
        'compressor_threshold': -20,
        'compressor_ratio': 3,
    }
    
    # Use provided values or defaults
    mix_orig = layer_mix_original if layer_mix_original is not None else defaults['layer_mix_original']
    mix_low = layer_mix_low if layer_mix_low is not None else defaults['layer_mix_low']
    mix_high = layer_mix_high if layer_mix_high is not None else defaults['layer_mix_high']
    mix_under = layer_mix_undertone if layer_mix_undertone is not None else defaults['layer_mix_undertone']
    
    # Normalize mix ratios
    total = mix_orig + mix_low + mix_high + mix_under
    if total > 0:
        mix_orig /= total
        mix_low /= total
        mix_high /= total
        mix_under /= total
    
    # Create layers
    layer1 = y.copy()
    layer2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1)
    delay_samples = int(0.030 * sr)
    layer2 = np.pad(layer2, (delay_samples, 0))[:len(y)]
    
    layer3 = librosa.effects.pitch_shift(y, sr=sr, n_steps=0.5)
    delay_samples = int(0.050 * sr)
    layer3 = np.pad(layer3, (delay_samples, 0))[:len(y)]
    
    layer4 = librosa.effects.pitch_shift(y, sr=sr, n_steps=-3)
    delay_samples = int(0.070 * sr)
    layer4 = np.pad(layer4, (delay_samples, 0))[:len(y)]
    
    # Mix with intensity scaling
    mixed = (mix_orig * layer1 + mix_low * layer2 + mix_high * layer3 + mix_under * layer4) * intensity
    mixed = mixed / (np.max(np.abs(mixed)) + 1e-8) * 0.85
    
    # Pedalboard with parameters
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=highpass_cutoff or defaults['highpass_cutoff']),
        Chorus(
            rate_hz=chorus_rate or defaults['chorus_rate'],
            depth=chorus_depth or defaults['chorus_depth'],
            mix=chorus_mix or defaults['chorus_mix']
        ),
        Reverb(
            room_size=reverb_room_size or defaults['reverb_room_size'],
            damping=reverb_damping or defaults['reverb_damping'],
            wet_level=reverb_wet_level or defaults['reverb_wet_level'],
            dry_level=reverb_dry_level or defaults['reverb_dry_level']
        ),
        Compressor(
            threshold_db=compressor_threshold or defaults['compressor_threshold'],
            ratio=compressor_ratio or defaults['compressor_ratio']
        ),
        Limiter(threshold_db=-1),
    ])
    
    processed = board(mixed.reshape(1, -1), sr)
    processed = processed.flatten()
    
    # Add mode-specific static
    static = _generate_static(len(processed), sr, "whisper_chorus", level=static_level_override)
    processed = processed + static
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.9
    
    return processed


def _filter_spore_cloud_param(
    y: np.ndarray, sr: int, intensity: float,
    reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
    chorus_rate, chorus_depth, chorus_mix,
    delay_time, delay_feedback, delay_mix,
    highpass_cutoff, lowpass_cutoff, compressor_threshold, compressor_ratio,
    grain_rate, pulse_frequency, time_stretch_rate,
    static_level_override
) -> np.ndarray:
    """Parameterized Spore Cloud filter."""
    defaults = {
        'grain_rate': 8.0,
        'pulse_frequency': 0.5,
        'time_stretch_rate': 1.02,
        'reverb_room_size': 0.7,
        'reverb_damping': 0.5,
        'reverb_wet_level': 0.45,
        'reverb_dry_level': 0.55,
        'chorus_rate': 0.2,
        'chorus_depth': 0.3,
        'chorus_mix': 0.3,
        'delay_time': 0.15,
        'delay_feedback': 0.2,
        'delay_mix': 0.15,
        'highpass_cutoff': 200,
        'lowpass_cutoff': 8000,
        'compressor_threshold': -18,
        'compressor_ratio': 4,
    }
    
    t = np.arange(len(y)) / sr
    
    # Granular envelope
    grain_rate_val = grain_rate or defaults['grain_rate']
    noise = np.random.randn(len(y))
    smoothed_noise = uniform_filter1d(noise, size=int(sr / grain_rate_val))
    grain_envelope = 0.65 + 0.35 * (smoothed_noise / (np.max(np.abs(smoothed_noise)) + 1e-8))
    
    # Pulse
    pulse_freq = pulse_frequency or defaults['pulse_frequency']
    slow_pulse = 0.85 + 0.15 * np.sin(2 * np.pi * pulse_freq * t)
    
    y_granular = y * grain_envelope * slow_pulse * intensity
    
    # Time stretch
    stretch_rate = time_stretch_rate or defaults['time_stretch_rate']
    y_stretched = librosa.effects.time_stretch(y_granular, rate=stretch_rate)
    if len(y_stretched) > len(y):
        y_granular = y_stretched[:len(y)]
    else:
        y_granular = np.pad(y_stretched, (0, len(y) - len(y_stretched)))
    
    y_granular = y_granular / (np.max(np.abs(y_granular)) + 1e-8) * 0.8
    
    # Pedalboard
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=highpass_cutoff or defaults['highpass_cutoff']),
        LowpassFilter(cutoff_frequency_hz=lowpass_cutoff or defaults['lowpass_cutoff']),
        Chorus(
            rate_hz=chorus_rate or defaults['chorus_rate'],
            depth=chorus_depth or defaults['chorus_depth'],
            mix=chorus_mix or defaults['chorus_mix']
        ),
        Reverb(
            room_size=reverb_room_size or defaults['reverb_room_size'],
            damping=reverb_damping or defaults['reverb_damping'],
            wet_level=reverb_wet_level or defaults['reverb_wet_level'],
            dry_level=reverb_dry_level or defaults['reverb_dry_level']
        ),
        Delay(
            delay_seconds=delay_time or defaults['delay_time'],
            feedback=delay_feedback or defaults['delay_feedback'],
            mix=delay_mix or defaults['delay_mix']
        ),
        Compressor(
            threshold_db=compressor_threshold or defaults['compressor_threshold'],
            ratio=compressor_ratio or defaults['compressor_ratio']
        ),
        Limiter(threshold_db=-1),
    ])
    
    processed = board(y_granular.reshape(1, -1), sr)
    processed = processed.flatten()
    
    # Add mode-specific static
    static = _generate_static(len(processed), sr, "spore_cloud", level=static_level_override)
    processed = processed + static
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.8
    
    return processed


def _filter_mycelium_pulse_param(
    y: np.ndarray, sr: int, intensity: float,
    reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
    highpass_cutoff, lowpass_cutoff, compressor_threshold, compressor_ratio,
    gain_db, pulse_rate, drone_frequency, drone_amplitude, pitch_shift,
    static_level_override
) -> np.ndarray:
    """Parameterized Mycelium Pulse filter."""
    defaults = {
        'pulse_rate': 0.7,
        'drone_frequency': 35.0,
        'drone_amplitude': 0.08,
        'pitch_shift': -2.0,
        'reverb_room_size': 0.6,
        'reverb_damping': 0.8,
        'reverb_wet_level': 0.35,
        'reverb_dry_level': 0.65,
        'highpass_cutoff': 60,
        'lowpass_cutoff': 6000,
        'compressor_threshold': -15,
        'compressor_ratio': 4,
        'gain_db': 2,
    }
    
    t = np.arange(len(y)) / sr
    
    # Pulse
    pulse_rate_val = pulse_rate or defaults['pulse_rate']
    pulse = 0.75 + 0.25 * np.sin(2 * np.pi * pulse_rate_val * t)
    slow_mod = 0.9 + 0.1 * np.sin(2 * np.pi * 0.15 * t)
    y_pulsed = y * pulse * slow_mod * intensity
    
    # Pitch shift
    pitch_val = pitch_shift or defaults['pitch_shift']
    y_deep = librosa.effects.pitch_shift(y_pulsed, sr=sr, n_steps=pitch_val)
    
    # Drone
    drone_freq = drone_frequency or defaults['drone_frequency']
    drone_amp = drone_amplitude or defaults['drone_amplitude']
    drone = drone_amp * np.sin(2 * np.pi * drone_freq * t)
    speech_envelope = np.abs(librosa.effects.preemphasis(y))
    speech_envelope = uniform_filter1d(speech_envelope, size=int(sr * 0.1))
    speech_envelope = speech_envelope / (np.max(speech_envelope) + 1e-8)
    drone = drone * speech_envelope
    
    y_mixed = y_deep + drone
    y_mixed = y_mixed / (np.max(np.abs(y_mixed)) + 1e-8) * 0.85
    
    # Pedalboard
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=highpass_cutoff or defaults['highpass_cutoff']),
        LowpassFilter(cutoff_frequency_hz=lowpass_cutoff or defaults['lowpass_cutoff']),
        Reverb(
            room_size=reverb_room_size or defaults['reverb_room_size'],
            damping=reverb_damping or defaults['reverb_damping'],
            wet_level=reverb_wet_level or defaults['reverb_wet_level'],
            dry_level=reverb_dry_level or defaults['reverb_dry_level']
        ),
        Compressor(
            threshold_db=compressor_threshold or defaults['compressor_threshold'],
            ratio=compressor_ratio or defaults['compressor_ratio']
        ),
        Gain(gain_db=gain_db or defaults['gain_db']),
        Limiter(threshold_db=-1),
    ])
    
    processed = board(y_mixed.reshape(1, -1), sr)
    processed = processed.flatten()
    
    # Add mode-specific static
    static = _generate_static(len(processed), sr, "mycelium_pulse", level=static_level_override)
    processed = processed + static
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.85
    
    return processed


def _filter_resonance_capture_param(
    y: np.ndarray, sr: int, intensity: float,
    reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
    highpass_cutoff, lowpass_cutoff, compressor_threshold, compressor_ratio,
    static_level, bitcrush_depth, echo_mix,
    static_level_override
) -> np.ndarray:
    """Parameterized Resonance Capture filter."""
    defaults = {
        'static_level': 0.03,
        'bitcrush_depth': 10,
        'echo_mix': 0.08,
        'reverb_room_size': 0.3,
        'reverb_damping': 0.9,
        'reverb_wet_level': 0.2,
        'reverb_dry_level': 0.8,
        'highpass_cutoff': 200,
        'lowpass_cutoff': 7000,
        'compressor_threshold': -18,
        'compressor_ratio': 5,
    }
    
    t = np.arange(len(y)) / sr
    
    # Dropout
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
    
    y_glitched = y * dropout_envelope * intensity
    
    # Echo
    echo_mix_val = echo_mix or defaults['echo_mix']
    echo_voice = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
    delay_samples = int(0.1 * sr)
    echo_voice = np.pad(echo_voice, (delay_samples, 0))[:len(y)]
    
    y_mixed = y_glitched + echo_mix_val * echo_voice
    y_mixed = y_mixed / (np.max(np.abs(y_mixed)) + 1e-8) * 0.9
    
    # Pedalboard
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=highpass_cutoff or defaults['highpass_cutoff']),
        Bitcrush(bit_depth=bitcrush_depth or defaults['bitcrush_depth']),
        LowpassFilter(cutoff_frequency_hz=lowpass_cutoff or defaults['lowpass_cutoff']),
        Reverb(
            room_size=reverb_room_size or defaults['reverb_room_size'],
            damping=reverb_damping or defaults['reverb_damping'],
            wet_level=reverb_wet_level or defaults['reverb_wet_level'],
            dry_level=reverb_dry_level or defaults['reverb_dry_level']
        ),
        Compressor(
            threshold_db=compressor_threshold or defaults['compressor_threshold'],
            ratio=compressor_ratio or defaults['compressor_ratio']
        ),
        Limiter(threshold_db=-1),
    ])
    
    processed = board(y_mixed.reshape(1, -1), sr)
    processed = processed.flatten()
    
    # Add mode-specific static (use static_level_override if provided, otherwise use static_level parameter)
    static_lev = static_level_override if static_level_override is not None else (static_level or defaults['static_level'])
    static = _generate_static(len(processed), sr, "resonance_capture", level=static_lev)
    processed = processed + static
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.9
    
    return processed


def _filter_transmission_param(
    y: np.ndarray, sr: int, intensity: float,
    reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
    highpass_cutoff, lowpass_cutoff, compressor_threshold, compressor_ratio,
    gain_db, pitch_shift,
    static_level_override
) -> np.ndarray:
    """Parameterized Transmission filter."""
    defaults = {
        'pitch_shift': -0.5,
        'reverb_room_size': 0.2,
        'reverb_damping': 0.8,
        'reverb_wet_level': 0.1,
        'reverb_dry_level': 0.9,
        'highpass_cutoff': 100,
        'lowpass_cutoff': 12000,
        'compressor_threshold': -18,
        'compressor_ratio': 3,
        'gain_db': 1,
    }
    
    # Pitch shift
    pitch_val = pitch_shift or defaults['pitch_shift']
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_val)
    y_blended = (0.7 * y + 0.3 * y_shifted) * intensity
    y_blended = y_blended / (np.max(np.abs(y_blended)) + 1e-8) * 0.9
    
    # Pedalboard
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=highpass_cutoff or defaults['highpass_cutoff']),
        LowpassFilter(cutoff_frequency_hz=lowpass_cutoff or defaults['lowpass_cutoff']),
        Compressor(
            threshold_db=compressor_threshold or defaults['compressor_threshold'],
            ratio=compressor_ratio or defaults['compressor_ratio']
        ),
        Reverb(
            room_size=reverb_room_size or defaults['reverb_room_size'],
            damping=reverb_damping or defaults['reverb_damping'],
            wet_level=reverb_wet_level or defaults['reverb_wet_level'],
            dry_level=reverb_dry_level or defaults['reverb_dry_level']
        ),
        Gain(gain_db=gain_db or defaults['gain_db']),
        Limiter(threshold_db=-0.5),
    ])
    
    processed = board(y_blended.reshape(1, -1), sr)
    processed = processed.flatten()
    
    # Add mode-specific static
    static = _generate_static(len(processed), sr, "transmission", level=static_level_override)
    processed = processed + static
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.9

    return processed


def _filter_cat_inner_voice_param(
    y: np.ndarray, sr: int, intensity: float,
    reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
    chorus_rate, chorus_depth, chorus_mix,
    highpass_cutoff, lowpass_cutoff, compressor_threshold, compressor_ratio,
    thought_pitch_shift, presence_dip, warmth_boost,
    cranial_reverb_size, thought_echo_mix, breathiness,
    static_level_override
) -> np.ndarray:
    """Parameterized Cat Inner Voice filter - female-optimized whispered inner monologue."""
    defaults = {
        'thought_pitch_shift': -0.3,
        'presence_dip': 0.40,
        'warmth_boost': 0.15,
        'cranial_reverb_size': 0.10,
        'thought_echo_mix': 0.0,
        'breathiness': 0.5,
        'reverb_room_size': 0.10,
        'reverb_damping': 0.88,
        'reverb_wet_level': 0.15,
        'reverb_dry_level': 0.85,
        'chorus_rate': 0.15,
        'chorus_depth': 0.08,
        'chorus_mix': 0.0,
        'highpass_cutoff': 120,
        'lowpass_cutoff': 7500,
        'compressor_threshold': -24,
        'compressor_ratio': 3.5,
    }

    # Use provided or defaults
    pitch_val = thought_pitch_shift if thought_pitch_shift is not None else defaults['thought_pitch_shift']
    presence_val = presence_dip if presence_dip is not None else defaults['presence_dip']
    warmth_val = warmth_boost if warmth_boost is not None else defaults['warmth_boost']
    cranial_size = cranial_reverb_size if cranial_reverb_size is not None else defaults['cranial_reverb_size']
    breath_val = breathiness if breathiness is not None else defaults['breathiness']

    # Use cranial_reverb_size to override reverb_room_size if reverb_room_size not explicitly set
    effective_reverb_room = reverb_room_size if reverb_room_size is not None else cranial_size

    # --- Pitch shift for thought quality ---
    y_thought = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_val)
    y_whisper = (0.15 * y + 0.85 * y_thought) * intensity

    # --- Very subtle saturation: closeness and density at low volume ---
    y_whisper = np.tanh(y_whisper * 1.2) * 0.85

    # --- EQ shaping: clean whisper through sculpting ---
    nyquist = sr / 2

    # Presence dip at 3.2 kHz - scaled by breathiness (more breathy = more dip)
    effective_presence = presence_val * (0.6 + breath_val * 0.8)
    if 3200 < nyquist and effective_presence > 0:
        low = max(0.01, min(0.99, (3200 - 800) / nyquist))
        high = max(low + 0.01, min(0.99, (3200 + 800) / nyquist))
        b, a = signal.butter(2, [low, high], 'band')
        presence_band = signal.filtfilt(b, a, y_whisper)
        y_whisper = y_whisper - effective_presence * presence_band

    # Sibilance dip (de-esser): tame harsh S sounds at 5-8 kHz
    if 6000 < nyquist:
        low_s = max(0.01, min(0.99, 5000 / nyquist))
        high_s = max(low_s + 0.01, min(0.99, 8000 / nyquist))
        b, a = signal.butter(2, [low_s, high_s], 'band')
        sibilance_band = signal.filtfilt(b, a, y_whisper)
        y_whisper = y_whisper - 0.30 * sibilance_band

    # Warmth boost around 800-1400 Hz
    if 1000 < nyquist and warmth_val > 0:
        low_w = max(0.01, min(0.99, 700 / nyquist))
        high_w = max(low_w + 0.01, min(0.99, 1400 / nyquist))
        b, a = signal.butter(2, [low_w, high_w], 'band')
        warmth_band = signal.filtfilt(b, a, y_whisper)
        y_whisper = y_whisper + warmth_val * warmth_band

    y_whisper = y_whisper / (np.max(np.abs(y_whisper)) + 1e-8) * 0.85

    # --- Pedalboard: clean whisper chain ---
    # Breathiness slider tightens the highpass and lowers the lowpass
    effective_hp = (highpass_cutoff or defaults['highpass_cutoff']) + (breath_val * 80)
    effective_lp = (lowpass_cutoff or defaults['lowpass_cutoff']) - (breath_val * 1500)
    effective_lp = max(4000, effective_lp)

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=effective_hp),
        LowpassFilter(cutoff_frequency_hz=effective_lp),
        Compressor(
            threshold_db=compressor_threshold or defaults['compressor_threshold'],
            ratio=compressor_ratio or defaults['compressor_ratio']
        ),
        Reverb(
            room_size=effective_reverb_room,
            damping=reverb_damping or defaults['reverb_damping'],
            wet_level=reverb_wet_level or defaults['reverb_wet_level'],
            dry_level=reverb_dry_level or defaults['reverb_dry_level']
        ),
        Gain(gain_db=-3),
        Limiter(threshold_db=-1),
    ])

    processed = board(y_whisper.reshape(1, -1), sr)
    processed = processed.flatten()

    # No static - clean whisper
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.75

    return processed


def _filter_cat_serious_param(
    y: np.ndarray, sr: int, intensity: float,
    reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level,
    highpass_cutoff, lowpass_cutoff, compressor_threshold, compressor_ratio,
    gain_db, seriousness, gravitas_boost, command_presence,
    static_level_override
) -> np.ndarray:
    """Parameterized Cat Serious filter - grades of seriousness."""
    defaults = {
        'seriousness': 0.5,
        'gravitas_boost': 0.20,
        'command_presence': 0.12,
        'reverb_room_size': 0.08,
        'reverb_damping': 0.9,
        'reverb_wet_level': 0.08,
        'reverb_dry_level': 0.92,
        'highpass_cutoff': 80,
        'lowpass_cutoff': 8000,
        'compressor_threshold': -20,
        'compressor_ratio': 4.5,
        'gain_db': 1.5,
    }

    serious_val = seriousness if seriousness is not None else defaults['seriousness']
    gravitas_val = gravitas_boost if gravitas_boost is not None else defaults['gravitas_boost']
    command_val = command_presence if command_presence is not None else defaults['command_presence']

    # Seriousness maps to pitch drop: 0=no drop, 0.5=-3 semi, 1.0=-5 semi
    pitch_drop = -1 + (serious_val * -4)  # Range: -1 to -5 semitones

    # --- Pitch drop ---
    y_serious = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_drop)
    y_blended = (0.10 * y + 0.90 * y_serious) * intensity
    y_blended = y_blended / (np.max(np.abs(y_blended)) + 1e-8) * 0.9

    # --- EQ shaping ---
    nyquist = sr / 2

    # Gravitas: low-mid boost (200-500 Hz), scaled by seriousness
    if 500 < nyquist and gravitas_val > 0:
        low = max(0.01, min(0.99, 200 / nyquist))
        high = max(low + 0.01, min(0.99, 500 / nyquist))
        b, a = signal.butter(2, [low, high], 'band')
        gravitas_band = signal.filtfilt(b, a, y_blended)
        y_blended = y_blended + gravitas_val * (0.5 + serious_val) * gravitas_band

    # Command presence: 2-3 kHz boost
    if 2500 < nyquist and command_val > 0:
        low_p = max(0.01, min(0.99, 2000 / nyquist))
        high_p = max(low_p + 0.01, min(0.99, 3000 / nyquist))
        b, a = signal.butter(2, [low_p, high_p], 'band')
        command_band = signal.filtfilt(b, a, y_blended)
        y_blended = y_blended + command_val * command_band

    y_blended = y_blended / (np.max(np.abs(y_blended)) + 1e-8) * 0.9

    # Lowpass gets tighter with seriousness (less brightness)
    effective_lowpass = lowpass_cutoff or defaults['lowpass_cutoff']
    effective_lowpass = effective_lowpass - (serious_val * 1500)  # Up to 1.5 kHz cut
    effective_lowpass = max(4000, effective_lowpass)

    # Compression ratio increases with seriousness
    effective_ratio = (compressor_ratio or defaults['compressor_ratio']) + (serious_val * 2)

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=highpass_cutoff or defaults['highpass_cutoff']),
        LowpassFilter(cutoff_frequency_hz=effective_lowpass),
        Compressor(
            threshold_db=compressor_threshold or defaults['compressor_threshold'],
            ratio=effective_ratio
        ),
        Reverb(
            room_size=reverb_room_size or defaults['reverb_room_size'],
            damping=reverb_damping or defaults['reverb_damping'],
            wet_level=reverb_wet_level or defaults['reverb_wet_level'],
            dry_level=reverb_dry_level or defaults['reverb_dry_level']
        ),
        Gain(gain_db=gain_db or defaults['gain_db']),
        Limiter(threshold_db=-0.5),
    ])

    processed = board(y_blended.reshape(1, -1), sr)
    processed = processed.flatten()

    static = _generate_static(len(processed), sr, "cat_serious", level=static_level_override)
    processed = processed + static
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.9

    return processed


def _filter_voice_comm_param(
    y: np.ndarray, sr: int, intensity: float,
    compressor_threshold, compressor_ratio, gain_db,
    radio_bandpass_low, radio_bandpass_high,
    saturation, helmet_resonance, click_volume,
    static_level_override
) -> np.ndarray:
    """Parameterized Voice Comm filter - helmet radio with PTT click."""
    defaults = {
        'radio_bandpass_low': 300,
        'radio_bandpass_high': 3400,
        'saturation': 1.8,
        'helmet_resonance': 0.06,
        'click_volume': 0.45,
        'compressor_threshold': -18,
        'compressor_ratio': 8,
        'gain_db': 2,
    }

    sat_val = saturation if saturation is not None else defaults['saturation']
    bp_low = radio_bandpass_low if radio_bandpass_low is not None else defaults['radio_bandpass_low']
    bp_high = radio_bandpass_high if radio_bandpass_high is not None else defaults['radio_bandpass_high']
    helmet_val = helmet_resonance if helmet_resonance is not None else defaults['helmet_resonance']
    click_vol = click_volume if click_volume is not None else defaults['click_volume']

    # --- Saturation: radio limiter character ---
    y_radio = np.tanh(y * sat_val * intensity) * 0.7

    # --- Mix static INTO the voice signal ---
    static = _generate_static(len(y_radio), sr, "voice_comm", level=static_level_override)
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

    # --- Pedalboard ---
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=bp_low),
        LowpassFilter(cutoff_frequency_hz=bp_high),
        Compressor(
            threshold_db=compressor_threshold or defaults['compressor_threshold'],
            ratio=compressor_ratio or defaults['compressor_ratio']
        ),
        Reverb(
            room_size=0.05,
            damping=0.95,
            wet_level=helmet_val,
            dry_level=1.0 - helmet_val
        ),
        Gain(gain_db=gain_db or defaults['gain_db']),
        Limiter(threshold_db=-1),
    ])

    processed = board(y_radio.reshape(1, -1), sr)
    processed = processed.flatten()
    processed = processed / (np.max(np.abs(processed)) + 1e-8) * 0.85

    # --- Noise gate / squelch: kill silence between speech ---
    gate_envelope = np.abs(processed)
    gate_envelope = uniform_filter1d(gate_envelope, size=int(sr * 0.02))
    gate_threshold = np.max(gate_envelope) * 0.04
    gate_mask = (gate_envelope > gate_threshold).astype(np.float32)
    gate_smooth = uniform_filter1d(gate_mask, size=int(sr * 0.03))
    gate_smooth = np.clip(gate_smooth * 3, 0, 1)
    processed = processed * gate_smooth

    # --- Trim leading silence so press click is right before voice ---
    abs_signal = np.abs(processed)
    trim_threshold = np.max(abs_signal) * 0.02
    first_voice = 0
    while first_voice < len(processed) and abs_signal[first_voice] < trim_threshold:
        first_voice += 1
    lead_samples = min(int(sr * 0.005), first_voice)
    processed = processed[max(0, first_voice - lead_samples):]

    # --- Trim trailing silence so release click is right at end of voice ---
    abs_signal = np.abs(processed)
    trim_threshold = np.max(abs_signal) * 0.02
    last_voice = len(processed) - 1
    while last_voice > 0 and abs_signal[last_voice] < trim_threshold:
        last_voice -= 1
    tail_samples = min(int(sr * 0.005), len(processed) - last_voice - 1)
    processed = processed[:last_voice + tail_samples + 1]

    # --- PTT press click at start, release click at end ---
    if click_vol > 0:
        press_click = _generate_ptt_click(sr, click_type="press")
        press_click = press_click * (click_vol / 0.85)  # Scale relative to default amp
        release_click = _generate_ptt_click(sr, click_type="release")
        release_click = release_click * (click_vol / 0.85)
        processed = np.concatenate([press_click, processed, release_click])

    return processed
