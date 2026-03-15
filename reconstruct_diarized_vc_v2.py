#!/usr/bin/env python3
"""
Reconstruct from improved VC conversion with proper voice switching.
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path

SPLITS_DIR = Path("Samples/diarized_splits")
VC_OUTPUT_DIR = Path("Samples/diarized_vc_output_v2")
FINAL_OUTPUT = Path("Samples/Autopsy_CAT_DANIEL_CORRECTED.wav")

print("Reconstructing audio from corrected VC conversion...")
print(f"Source: {VC_OUTPUT_DIR}")
print(f"Output: {FINAL_OUTPUT}\n")

# Load segments in order
audio_segments = []
sample_rate = 24000
loaded = 0
missing = 0

for orig_file in sorted(SPLITS_DIR.glob("*.wav")):
    speaker = "FEMALE" if "FEMALE" in orig_file.name else "MALE"
    seg_num = orig_file.name.split("_")[1].split(".")[0]
    converted_name = f"{speaker}_converted_{seg_num}.wav"
    converted_path = VC_OUTPUT_DIR / converted_name

    if converted_path.exists():
        try:
            audio_data, sr = sf.read(str(converted_path))
            audio_segments.append(audio_data)
            loaded += 1

            if loaded % 30 == 0:
                print(f"  [{loaded}/{len(list(SPLITS_DIR.glob('*.wav')))}] Loaded {converted_name}")

        except Exception as e:
            print(f"  ERROR loading {converted_name}: {e}")
            missing += 1
    else:
        missing += 1

print(f"\nConcatenating {len(audio_segments)} segments...")
final_audio = np.concatenate(audio_segments, axis=0)

print(f"Exporting to {FINAL_OUTPUT}...")
sf.write(str(FINAL_OUTPUT), final_audio, sample_rate)

duration_min = len(final_audio) / sample_rate / 60
file_size_mb = FINAL_OUTPUT.stat().st_size / (1024 * 1024)

print(f"\n" + "="*70)
print("CORRECTED RECONSTRUCTION COMPLETE!")
print("="*70)
print(f"Duration: {duration_min:.2f} minutes")
print(f"File size: {file_size_mb:.1f} MB")
print(f"Loaded: {loaded}, Missing: {missing}")
print(f"\nVoices:")
print(f"  CAT (Female): Cat Mitch Eng.m4a (180.6 Hz)")
print(f"  DANIEL (Male): daniel_obrien_reference.wav (100.8 Hz)")
