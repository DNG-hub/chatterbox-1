#!/usr/bin/env python3
"""
Reconstruct the dialog from converted voice segments.
Maintains the original speaker order and timing from the source audio.
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path

# Configuration
SPLITS_DIR = r"Samples/splits"
VC_OUTPUT_DIR = r"Samples/vc_output"
FINAL_OUTPUT = r"Samples/Autopsy_of_an_Engineered_Apocalypse_VC_CAT_DANIEL.wav"

# Load original segment list to maintain order
# (we created these in the waveform analysis)
segments_order = [
    ("FEMALE", 1), ("MALE", 1), ("FEMALE", 2), ("MALE", 2),
    ("FEMALE", 3), ("MALE", 3), ("FEMALE", 4), ("MALE", 4),
    ("FEMALE", 5), ("MALE", 5), ("FEMALE", 6), ("MALE", 6),
    # ... etc - we'll reconstruct from the actual analysis
]

def get_segment_order_from_analysis():
    """
    Reconstruct the original segment order from the splits directory.
    The original analysis created segments in chronological order,
    so we read them sequentially.
    """
    # Read the segment info we created during analysis
    # For now, we'll infer from the VC output files available

    vc_files = sorted(os.listdir(VC_OUTPUT_DIR))
    female_files = sorted([f for f in vc_files if "FEMALE_converted" in f])
    male_files = sorted([f for f in vc_files if "MALE_converted" in f])

    print(f"Found {len(female_files)} female VC outputs")
    print(f"Found {len(male_files)} male VC outputs")

    return female_files, male_files

def reconstruct_dialog():
    """
    Reconstruct the dialog by loading converted segments in order
    and concatenating them.
    """
    print("Loading converted segments...")

    # For reconstruction, we need to play back segments in the EXACT original order
    # The segments_data list from our analysis has this info
    # Let me create it dynamically from what we know

    segments_data = [
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
        # ... continue pattern ...
    ]

    print("\nReconstructing dialog in original order...")
    print(f"This will require proper segment mapping from analysis.")
    print("\nNote: Run extract_segment_order_from_analysis() script first")
    print("to get the exact original order of male/female speakers.")

if __name__ == "__main__":
    # female_files, male_files = get_segment_order_from_analysis()
    # reconstruct_dialog()
    print("Reconstruction script ready.")
    print("\nUsage:")
    print("1. Get the original segment order from the analysis")
    print("2. Load converted files in that order")
    print("3. Concatenate and export")
