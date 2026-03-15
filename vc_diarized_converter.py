#!/usr/bin/env python3
"""
Voice conversion for diarized segments with automatic speaker-to-voice mapping.
Run this with: python vc_diarized_converter.py
"""

import os
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import soundfile as sf
from chatterbox.vc import ChatterboxVC

# Configuration
SPLITS_DIR = Path("Samples/diarized_splits")
OUTPUT_DIR = Path("Samples/diarized_vc_output")
CAT_REF = Path(".gradio/flagged/Target voice audio file if none, the default voice is used/5e758f958b23dda44386/Cat Mitch Eng.m4a")
DANIEL_REF = Path("E:\\REPOS\\StoryTeller\\docs\\voices\\Dav_Daniel.wav")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def convert_segments():
    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("\nLoading Chatterbox Voice Conversion model...")
    vc = ChatterboxVC.from_pretrained(device=device)

    # Get segments
    split_files = sorted([f for f in SPLITS_DIR.glob("*.wav")])
    female_splits = sorted([f for f in split_files if "FEMALE" in f.name])
    male_splits = sorted([f for f in split_files if "MALE" in f.name])

    print(f"Found {len(female_splits)} female + {len(male_splits)} male segments")

    # Process Female -> CAT
    print("\n" + "="*70)
    print("CONVERTING FEMALE SEGMENTS TO CAT VOICE")
    print("="*70)

    female_count = 0
    for i, fpath in enumerate(female_splits, 1):
        try:
            print(f"[{i}/{len(female_splits)}] {fpath.name}...", end=" ", flush=True)

            vc.set_target_voice(str(CAT_REF))
            vc_audio = vc.generate(str(fpath))

            output_name = fpath.name.replace("FEMALE", "FEMALE_converted")
            output_path = OUTPUT_DIR / output_name

            vc_audio_np = vc_audio.squeeze(0).numpy()
            sf.write(str(output_path), vc_audio_np, vc.sr)

            duration = len(vc_audio_np) / vc.sr
            print(f"OK ({duration:.2f}s)")
            female_count += 1

        except Exception as e:
            print(f"ERROR: {str(e)[:60]}")

    # Process Male -> Daniel
    print("\n" + "="*70)
    print("CONVERTING MALE SEGMENTS TO DANIEL VOICE")
    print("="*70)

    male_count = 0
    for i, fpath in enumerate(male_splits, 1):
        try:
            print(f"[{i}/{len(male_splits)}] {fpath.name}...", end=" ", flush=True)

            vc.set_target_voice(str(DANIEL_REF))
            vc_audio = vc.generate(str(fpath))

            output_name = fpath.name.replace("MALE", "MALE_converted")
            output_path = OUTPUT_DIR / output_name

            vc_audio_np = vc_audio.squeeze(0).numpy()
            sf.write(str(output_path), vc_audio_np, vc.sr)

            duration = len(vc_audio_np) / vc.sr
            print(f"OK ({duration:.2f}s)")
            male_count += 1

        except Exception as e:
            print(f"ERROR: {str(e)[:60]}")

    print(f"\n" + "="*70)
    print("CONVERSION COMPLETE")
    print("="*70)
    print(f"Female: {female_count}/{len(female_splits)}")
    print(f"Male: {male_count}/{len(male_splits)}")
    print(f"Total: {female_count + male_count}/{len(split_files)}")

if __name__ == "__main__":
    convert_segments()
