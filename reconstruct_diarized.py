#!/usr/bin/env python3
"""
Reconstruct full audio from diarized segments (original voices).
This preserves the full 18-minute duration with natural speaker boundaries.
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path

# Configuration
SPLITS_DIR = Path("Samples/diarized_splits")
FINAL_OUTPUT = Path("Samples/Autopsy_of_an_Engineered_Apocalypse_DIARIZED.wav")

print("Reconstructing audio from diarized segments...")
print(f"Output: {FINAL_OUTPUT}\n")

# Get segments in order (they're named sequentially: FEMALE_001, FEMALE_002, etc.)
female_files = sorted(SPLITS_DIR.glob("FEMALE_*.wav"))
male_files = sorted(SPLITS_DIR.glob("MALE_*.wav"))

all_files = sorted(SPLITS_DIR.glob("*.wav"))

print(f"Found {len(female_files)} female + {len(male_files)} male = {len(all_files)} total segments\n")

# Load and concatenate in the order they appear in directory (chronological)
audio_segments = []
sample_rate = 44100
total_duration = 0

for i, fpath in enumerate(all_files, 1):
    try:
        audio_data, sr = sf.read(str(fpath))
        audio_segments.append(audio_data)
        duration = len(audio_data) / sr
        total_duration += duration

        if i % 30 == 0:
            print(f"[{i}/{len(all_files)}] Loaded {fpath.name} ({duration:.2f}s, cumulative: {total_duration:.1f}s)")

    except Exception as e:
        print(f"ERROR loading {fpath.name}: {e}")

print(f"\nConcatenating {len(audio_segments)} segments...")
final_audio = np.concatenate(audio_segments, axis=0)

print(f"Exporting to {FINAL_OUTPUT}...")
sf.write(str(FINAL_OUTPUT), final_audio, sample_rate)

final_duration_sec = len(final_audio) / sample_rate
final_duration_min = final_duration_sec / 60
file_size_mb = os.path.getsize(str(FINAL_OUTPUT)) / (1024 * 1024)

print(f"\nReconstruction complete!")
print(f"Output duration: {final_duration_min:.2f} minutes ({final_duration_sec:.0f}s)")
print(f"Output size: {file_size_mb:.1f} MB")
print(f"Sample rate: {sample_rate} Hz")
print(f"\nNOTE: This uses original voices. For CAT+Daniel conversion,")
print(f"the VC model needs to be applied to each segment.")
