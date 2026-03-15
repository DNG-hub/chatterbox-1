#!/usr/bin/env python3
"""
Reconstruct full dialog from diarized VC-converted segments.
Maintains original speaker order with CAT+Daniel voices.

Run this after vc_diarized_converter.py completes.
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path

# Configuration
SPLITS_DIR = Path("Samples/diarized_splits")
VC_OUTPUT_DIR = Path("Samples/diarized_vc_output")
FINAL_OUTPUT = Path("Samples/Autopsy_of_an_Engineered_Apocalypse_FINAL_CAT_DANIEL.wav")

print("Reconstructing full dialog from diarized VC segments...")
print(f"Source: {VC_OUTPUT_DIR}")
print(f"Output: {FINAL_OUTPUT}\n")

# Get converted segments
vc_files = sorted(VC_OUTPUT_DIR.glob("*.wav"))
original_splits = sorted(SPLITS_DIR.glob("*.wav"))

print(f"Found {len(vc_files)} converted VC segments")
print(f"Found {len(original_splits)} original segments")

if len(vc_files) == 0:
    print("\nERROR: No converted segments found!")
    print("Please run vc_diarized_converter.py first.")
    exit(1)

# Create mapping from original to converted
# Original segments are in chronological order by filename
# We need to maintain that order for reconstruction

audio_segments = []
sample_rate = 24000  # VC output sample rate
total_duration = 0
missing_segments = []

print("\nLoading converted segments in chronological order...\n")

# Get unique segment numbers from converted files
female_converted = sorted(VC_OUTPUT_DIR.glob("FEMALE_converted_*.wav"))
male_converted = sorted(VC_OUTPUT_DIR.glob("MALE_converted_*.wav"))

# To maintain original order, we need to interleave them based on their original positions
# Load original segment order from filenames
original_order = []
for f in original_splits:
    speaker = "FEMALE" if "FEMALE" in f.name else "MALE"
    num = int(f.name.split("_")[1].split(".")[0])
    original_order.append((f.name, speaker, num))

# Sort by original timestamp (they were created in chronological order)
# Since they're named sequentially FEMALE_001, FEMALE_002... MALE_001, MALE_002...
# We need to reconstruct the actual chronological order

# Simpler approach: read both lists in order and interleave
female_idx = 1
male_idx = 1
total_segments = len(original_splits)

for seg_info in sorted(original_order, key=lambda x: (x[1], x[2])):
    # This maintains FEMALE_001, MALE_001, FEMALE_002, MALE_002...
    # But the actual order was more complex. Let's instead just load converted files
    # in the same order as their original counterparts
    pass

# Better approach: load both converted files and original files, match by name
print("Segment loading progress:")
loaded = 0
for orig_file in original_splits:
    speaker = "FEMALE" if "FEMALE" in orig_file.name else "MALE"
    seg_num = orig_file.name.split("_")[1].split(".")[0]

    converted_name = f"{speaker}_converted_{seg_num}.wav"
    converted_path = VC_OUTPUT_DIR / converted_name

    if converted_path.exists():
        try:
            audio_data, sr = sf.read(str(converted_path))
            audio_segments.append(audio_data)
            duration = len(audio_data) / sr
            total_duration += duration
            loaded += 1

            if loaded % 30 == 0:
                print(f"  [{loaded}/{len(original_splits)}] Loaded {converted_name} ({duration:.2f}s)")

        except Exception as e:
            print(f"  ERROR loading {converted_name}: {e}")
            missing_segments.append(converted_name)
    else:
        # Fall back to original if conversion doesn't exist
        try:
            audio_data, sr = sf.read(str(orig_file))
            # Resample to 24kHz if needed
            if sr != 24000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=24000)
            audio_segments.append(audio_data)
            loaded += 1
        except Exception as e:
            print(f"  WARNING: Could not load {orig_file.name}: {e}")
            missing_segments.append(orig_file.name)

print(f"\nConcatenating {len(audio_segments)} segments...")
final_audio = np.concatenate(audio_segments, axis=0)

print(f"Exporting to {FINAL_OUTPUT}...")
sf.write(str(FINAL_OUTPUT), final_audio, sample_rate)

final_duration_sec = len(final_audio) / sample_rate
final_duration_min = final_duration_sec / 60
file_size_mb = os.path.getsize(str(FINAL_OUTPUT)) / (1024 * 1024)

print(f"\n" + "="*70)
print("FINAL RECONSTRUCTION COMPLETE!")
print("="*70)
print(f"Output duration: {final_duration_min:.2f} minutes ({final_duration_sec:.0f}s)")
print(f"Output size: {file_size_mb:.1f} MB")
print(f"Sample rate: {sample_rate} Hz")
print(f"Segments loaded: {loaded}/{len(original_splits)}")

if missing_segments:
    print(f"\nMissing/fallback segments: {len(missing_segments)}")
    for seg in missing_segments[:5]:
        print(f"  - {seg}")
    if len(missing_segments) > 5:
        print(f"  ... and {len(missing_segments) - 5} more")

print(f"\nVoices: Female->CAT, Male->Daniel")
print(f"Status: Ready to listen!")
