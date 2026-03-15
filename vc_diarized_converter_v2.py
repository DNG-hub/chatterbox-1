#!/usr/bin/env python3
"""
Voice conversion for diarized segments with PROPER voice switching.
Uses best reference voices for CAT and DANIEL.
"""

import os
import torch
import soundfile as sf
from src.chatterbox.vc import ChatterboxVC

# Configuration - BEST REFERENCES
SPLITS_DIR = r"Samples/diarized_splits"
OUTPUT_DIR = r"Samples/diarized_vc_output_v2"
CAT_REF = r"E:\REPOS\StoryTeller\docs\voices\Cat Mitch Eng.m4a"  # Female: 180.6 Hz
DANIEL_REF = r"E:\REPOS\StoryTeller\docs\voices\daniel_obrien_reference.wav"  # Male: 100.8 Hz

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device detection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
print("\nLoading Chatterbox Voice Conversion model...")
vc = ChatterboxVC.from_pretrained(device=device)

# Get segments
split_files = sorted([f for f in os.listdir(SPLITS_DIR) if f.endswith('.wav')])
female_splits = sorted([f for f in split_files if "FEMALE" in f])
male_splits = sorted([f for f in split_files if "MALE" in f])

print(f"Found {len(female_splits)} female + {len(male_splits)} male segments")

# Convert Female -> CAT
print("\n" + "="*70)
print(f"CONVERTING FEMALE SEGMENTS TO CAT VOICE")
print(f"Reference: Cat Mitch Eng.m4a (pitch: 180.6 Hz)")
print("="*70)

female_converted = 0
for i, fname in enumerate(female_splits, 1):
    input_path = os.path.join(SPLITS_DIR, fname)
    segment_num = fname.replace("FEMALE_", "").replace(".wav", "").zfill(3)
    output_fname = f"FEMALE_converted_{segment_num}.wav"
    output_path = os.path.join(OUTPUT_DIR, output_fname)

    try:
        print(f"[{i}/{len(female_splits)}] {fname}...", end=" ", flush=True)

        # SET CAT VOICE REFERENCE
        vc.set_target_voice(CAT_REF)
        # GENERATE
        vc_audio = vc.generate(input_path)

        # Save
        vc_audio_np = vc_audio.squeeze(0).numpy()
        sf.write(output_path, vc_audio_np, vc.sr)

        female_converted += 1
        duration = len(vc_audio_np) / vc.sr
        print(f"OK ({duration:.2f}s)")

    except Exception as e:
        print(f"ERROR: {str(e)[:60]}")

# Convert Male -> DANIEL
print("\n" + "="*70)
print(f"CONVERTING MALE SEGMENTS TO DANIEL VOICE")
print(f"Reference: daniel_obrien_reference.wav (pitch: 100.8 Hz)")
print("="*70)

male_converted = 0
for i, fname in enumerate(male_splits, 1):
    input_path = os.path.join(SPLITS_DIR, fname)
    segment_num = fname.replace("MALE_", "").replace(".wav", "").zfill(3)
    output_fname = f"MALE_converted_{segment_num}.wav"
    output_path = os.path.join(OUTPUT_DIR, output_fname)

    try:
        print(f"[{i}/{len(male_splits)}] {fname}...", end=" ", flush=True)

        # SET DANIEL VOICE REFERENCE
        vc.set_target_voice(DANIEL_REF)
        # GENERATE
        vc_audio = vc.generate(input_path)

        # Save
        vc_audio_np = vc_audio.squeeze(0).numpy()
        sf.write(output_path, vc_audio_np, vc.sr)

        male_converted += 1
        duration = len(vc_audio_np) / vc.sr
        print(f"OK ({duration:.2f}s)")

    except Exception as e:
        print(f"ERROR: {str(e)[:60]}")

print(f"\n" + "="*70)
print("CONVERSION SUMMARY")
print("="*70)
print(f"Female (CAT): {female_converted}/{len(female_splits)}")
print(f"Male (DANIEL): {male_converted}/{len(male_splits)}")
print(f"Total: {female_converted + male_converted}/{len(split_files)}")
print(f"\nOutput directory: {OUTPUT_DIR}")
