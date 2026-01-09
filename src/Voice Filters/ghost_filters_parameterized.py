"""
Parameterized Ghost Voice Filters - Enhanced version with adjustable parameters
Extends ghost_filters.py to support real-time parameter tweaking
"""

import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d
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
        get_mode_names,
        _generate_static
    )
except ImportError:
    # Fallback for direct import
    from ghost_filters import (
        apply_ghost_filter as base_apply_ghost_filter,
        GHOST_MODES,
        get_mode_names,
        _generate_static
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
