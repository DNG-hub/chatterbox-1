#!/usr/bin/env python3
"""
Reconstruct the full dialog from converted voice segments.
Maintains the original speaker order from the waveform analysis.
"""

import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# Configuration
VC_OUTPUT_DIR = r"Samples/vc_output"
FINAL_OUTPUT = r"Samples/Autopsy_of_an_Engineered_Apocalypse_CAT_DANIEL.wav"

# Original segment order (speaker, segment_number) from waveform analysis
# This is the exact chronological order from the source audio
original_segment_order = [
    ("FEMALE", 1), ("MALE", 1), ("FEMALE", 2), ("MALE", 2),
    ("FEMALE", 3), ("MALE", 3), ("FEMALE", 4), ("MALE", 4),
    ("FEMALE", 5), ("MALE", 5), ("FEMALE", 6), ("MALE", 6),
    ("FEMALE", 7), ("MALE", 7), ("FEMALE", 8), ("MALE", 8),
    ("FEMALE", 9), ("MALE", 9), ("FEMALE", 10), ("MALE", 10),
    ("FEMALE", 11), ("MALE", 11), ("FEMALE", 12), ("MALE", 12),
    ("FEMALE", 13), ("MALE", 13), ("FEMALE", 14), ("MALE", 14),
    ("FEMALE", 15), ("MALE", 15), ("FEMALE", 16), ("MALE", 16),
    ("FEMALE", 17), ("MALE", 17), ("FEMALE", 18), ("MALE", 18),
    ("FEMALE", 19), ("MALE", 19), ("FEMALE", 20), ("MALE", 20),
    ("FEMALE", 21), ("MALE", 21), ("FEMALE", 22), ("MALE", 22),
    ("FEMALE", 23), ("MALE", 23), ("FEMALE", 24), ("MALE", 24),
    ("FEMALE", 25), ("MALE", 25), ("FEMALE", 26), ("MALE", 26),
    ("FEMALE", 27), ("MALE", 27), ("FEMALE", 28), ("MALE", 28),
    ("FEMALE", 29), ("MALE", 29), ("FEMALE", 30), ("MALE", 30),
    ("FEMALE", 31), ("MALE", 31), ("FEMALE", 32), ("MALE", 32),
    ("FEMALE", 33), ("MALE", 33), ("FEMALE", 34), ("MALE", 34),
    ("FEMALE", 35), ("MALE", 35), ("FEMALE", 36), ("MALE", 36),
    ("FEMALE", 37), ("MALE", 37), ("FEMALE", 38), ("MALE", 38),
    ("FEMALE", 39), ("MALE", 39), ("FEMALE", 40), ("MALE", 40),
    ("FEMALE", 41), ("MALE", 41), ("FEMALE", 42), ("MALE", 42),
    ("FEMALE", 43), ("MALE", 43), ("FEMALE", 44), ("MALE", 44),
    ("FEMALE", 45), ("MALE", 45), ("FEMALE", 46), ("MALE", 46),
    ("FEMALE", 47), ("MALE", 47), ("FEMALE", 48), ("MALE", 48),
    ("FEMALE", 49), ("MALE", 49), ("FEMALE", 50), ("MALE", 50),
    ("FEMALE", 51), ("MALE", 51), ("FEMALE", 52), ("MALE", 52),
    ("FEMALE", 53), ("MALE", 53), ("FEMALE", 54), ("MALE", 54),
    ("FEMALE", 55), ("MALE", 55), ("FEMALE", 56), ("MALE", 56),
    ("FEMALE", 57), ("MALE", 57), ("FEMALE", 58), ("MALE", 58),
    ("FEMALE", 59), ("MALE", 59), ("FEMALE", 60), ("MALE", 60),
    ("FEMALE", 61), ("MALE", 61), ("FEMALE", 62), ("MALE", 62),
    ("FEMALE", 63), ("MALE", 63), ("FEMALE", 64), ("MALE", 64),
    ("FEMALE", 65), ("MALE", 65), ("FEMALE", 66), ("MALE", 66),
    ("FEMALE", 67), ("MALE", 67), ("FEMALE", 68), ("MALE", 68),
    ("FEMALE", 69), ("MALE", 69), ("FEMALE", 70), ("MALE", 70),
    ("FEMALE", 71), ("MALE", 71), ("FEMALE", 72), ("MALE", 72),
    ("FEMALE", 73), ("MALE", 73), ("FEMALE", 74), ("MALE", 74),
    ("FEMALE", 75), ("MALE", 75), ("FEMALE", 76), ("MALE", 76),
    ("FEMALE", 77), ("MALE", 77), ("FEMALE", 78), ("MALE", 78),
    ("FEMALE", 79), ("MALE", 79), ("FEMALE", 80), ("MALE", 80),
    ("FEMALE", 81), ("MALE", 81), ("FEMALE", 82), ("MALE", 82),
    ("FEMALE", 83), ("MALE", 83), ("FEMALE", 84), ("MALE", 84),
    ("FEMALE", 85), ("MALE", 85), ("FEMALE", 86), ("MALE", 86),
    ("FEMALE", 87), ("MALE", 87), ("FEMALE", 88), ("MALE", 88),
    ("FEMALE", 89), ("MALE", 89), ("FEMALE", 90), ("MALE", 90),
    ("FEMALE", 91), ("MALE", 91), ("FEMALE", 92), ("MALE", 92),
    ("FEMALE", 93), ("MALE", 93), ("FEMALE", 94), ("MALE", 94),
    ("FEMALE", 95), ("MALE", 95), ("FEMALE", 96), ("MALE", 96),
    ("FEMALE", 97), ("MALE", 97), ("FEMALE", 98), ("MALE", 98),
    ("FEMALE", 99), ("MALE", 99), ("FEMALE", 100), ("MALE", 100),
    ("FEMALE", 101), ("MALE", 101), ("FEMALE", 102), ("MALE", 102),
    ("FEMALE", 103), ("MALE", 103), ("FEMALE", 104), ("MALE", 104),
    ("FEMALE", 105), ("MALE", 105), ("FEMALE", 106), ("MALE", 106),
    ("FEMALE", 107), ("MALE", 107), ("FEMALE", 108), ("MALE", 108),
    ("FEMALE", 109), ("MALE", 109), ("FEMALE", 110), ("MALE", 110),
    ("FEMALE", 111), ("MALE", 111), ("FEMALE", 112), ("MALE", 112),
    ("FEMALE", 113), ("MALE", 113), ("FEMALE", 114), ("MALE", 114),
    ("FEMALE", 115), ("MALE", 115), ("FEMALE", 116), ("MALE", 116),
    ("FEMALE", 117), ("MALE", 117), ("FEMALE", 118), ("MALE", 118),
    ("FEMALE", 119), ("MALE", 119), ("FEMALE", 120), ("MALE", 120),
    ("FEMALE", 121), ("MALE", 121), ("FEMALE", 122), ("MALE", 122),
    ("FEMALE", 123), ("MALE", 123), ("FEMALE", 124), ("MALE", 124),
    ("FEMALE", 125), ("MALE", 125), ("FEMALE", 126), ("MALE", 126),
    ("FEMALE", 127), ("MALE", 127), ("FEMALE", 128), ("MALE", 128),
    ("FEMALE", 129), ("MALE", 129), ("FEMALE", 130), ("MALE", 130),
    ("FEMALE", 131), ("MALE", 131), ("FEMALE", 132), ("MALE", 132),
    ("FEMALE", 133),
]

def reconstruct_dialog():
    """
    Load converted segments in original order and concatenate.
    Falls back to original segments for empty/silent sections (pauses).
    """
    print("Reconstructing dialog from converted segments...")
    print(f"Target output: {FINAL_OUTPUT}\n")

    audio_segments = []
    sample_rate = 24000  # Chatterbox VC output sample rate
    SPLITS_DIR = r"Samples/splits"

    conversions_used = 0
    fallbacks_used = 0

    for i, (speaker, seg_num) in enumerate(original_segment_order, 1):
        # Format segment filename
        seg_num_str = str(seg_num).zfill(3)
        converted_filename = f"{speaker}_converted_{seg_num_str}.wav"
        converted_filepath = os.path.join(VC_OUTPUT_DIR, converted_filename)

        original_filename = f"{speaker}_{seg_num_str}.wav"
        original_filepath = os.path.join(SPLITS_DIR, original_filename)

        audio_data = None

        # Try to load converted version first
        if os.path.exists(converted_filepath):
            try:
                audio_data, sr = sf.read(converted_filepath)
                conversions_used += 1
            except Exception as e:
                print(f"ERROR loading converted {converted_filename}: {e}")

        # Fall back to original (silence/pause) if conversion failed or doesn't exist
        if audio_data is None and os.path.exists(original_filepath):
            try:
                audio_data, sr = sf.read(original_filepath)
                # Resample original to match output sample rate if needed
                if sr != sample_rate:
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)
                fallbacks_used += 1
            except Exception as e:
                print(f"ERROR loading original {original_filename}: {e}")

        if audio_data is not None:
            audio_segments.append(audio_data)

            if i % 30 == 0:
                duration = len(audio_data) / sample_rate
                print(f"[{i}/{len(original_segment_order)}] Loaded {speaker}_{seg_num_str} ({duration:.2f}s)")

    # Concatenate all segments
    print(f"\nConcatenating {len(audio_segments)} segments...")
    print(f"  - Converted segments: {conversions_used}")
    print(f"  - Original (silence/pause) segments: {fallbacks_used}")

    final_audio = np.concatenate(audio_segments, axis=0)

    # Export final audio
    print(f"\nExporting to {FINAL_OUTPUT}")
    sf.write(FINAL_OUTPUT, final_audio, sample_rate)

    duration_sec = len(final_audio) / sample_rate
    duration_min = duration_sec / 60
    print(f"\nReconstruction complete!")
    print(f"Output duration: {duration_min:.2f} minutes ({duration_sec:.0f}s)")
    print(f"Output size: {os.path.getsize(FINAL_OUTPUT) / (1024*1024):.1f} MB")
    print(f"Sample rate: {sample_rate} Hz")

if __name__ == "__main__":
    reconstruct_dialog()
