#!/usr/bin/env python3
"""
Extract male and female voice reference clips from a podcast audio file.
Uses speaker-annotated SRT to identify segments, then extracts the longest
clean solo segments suitable for TTS voice cloning reference.

SRT format:
  - [Male]/[Female] tags mark alternating speakers (original section, ends with [End])
  - [male]...[/male] tags mark additional male-only segments (ends with [end])

Output goes to storyteller/doc/voices/raw_extracts/ for review before finalization.
"""

import re
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
SOURCE_AUDIO = Path(r"E:\DaVinci_Projects\Cat_Daniel_Collapse_Protocol\Youtube\Podcast\Collapse Protocol Prolog\VO\Why_the_world_is_collapsing_on_purpose.m4a")
SOURCE_SRT = Path(r"E:\DaVinci_Projects\Cat_Daniel_Collapse_Protocol\Youtube\Podcast\Collapse Protocol Prolog\VO\PrologPodcast.srt")
OUT_DIR = Path(r"E:\DaVinci_Projects\Cat_Daniel_Collapse_Protocol\storyteller\doc\voices\raw_extracts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_SEGMENT_SECONDS = 5.0
MAX_CLIPS_PER_SPEAKER = 5
TARGET_SR = 24000  # S3GEN_SR


def ts_to_sec(t):
    h, m, s = t.replace(",", ".").split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


def parse_srt(srt_path):
    """Parse SRT and return (male_segments, female_segments) as lists of (start, end)."""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    entries = re.findall(
        r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)",
        content, re.DOTALL,
    )

    male_segs = []
    female_segs = []

    # --- Pass 1: Original uppercase alternating section (up to [End]) ---
    current = None
    seg_start = None
    for idx, st, et, text in entries:
        text_c = re.sub(r"</?b>", "", text).strip()
        if "[End]" in text_c:
            if current and seg_start is not None:
                target = male_segs if current == "Male" else female_segs
                target.append((seg_start, ts_to_sec(et)))
            break
        m = re.match(r"\[(Male|Female)\]", text_c)
        if m:
            new_spk = m.group(1)
            if current and seg_start is not None:
                target = male_segs if current == "Male" else female_segs
                target.append((seg_start, ts_to_sec(st)))
            current = new_spk
            seg_start = ts_to_sec(st)

    # --- Pass 2: New lowercase [male]...[/male] sections ---
    in_male = False
    male_start = None
    for idx, st, et, text_raw in entries:
        text_c = re.sub(r"</?b>", "", text_raw).strip()
        if "[male]" in text_c and "[/male]" not in text_c:
            in_male = True
            male_start = ts_to_sec(st)
        if "[/male]" in text_c:
            if "[male]" in text_c and male_start is None:
                male_start = ts_to_sec(st)
            male_segs.append((male_start, ts_to_sec(et)))
            in_male = False
            male_start = None

    return male_segs, female_segs


# ── Parse SRT ──────────────────────────────────────────────────────────────
print(f"Parsing: {SOURCE_SRT.name}")
male_segs, female_segs = parse_srt(SOURCE_SRT)
print(f"  Male segments:   {len(male_segs)} ({sum(e-s for s,e in male_segs):.1f}s total)")
print(f"  Female segments: {len(female_segs)} ({sum(e-s for s,e in female_segs):.1f}s total)")

# ── Load audio ─────────────────────────────────────────────────────────────
print(f"\nLoading: {SOURCE_AUDIO.name}")
audio_full, sr_orig = librosa.load(str(SOURCE_AUDIO), sr=None, mono=True)
duration_min = len(audio_full) / sr_orig / 60
print(f"  Duration: {duration_min:.1f} min, SR: {sr_orig} Hz")

# SRT timestamps start at 01:00:00 — subtract 1 hour offset
OFFSET = 3600.0

# Resample to target SR
print(f"  Resampling {sr_orig} -> {TARGET_SR} Hz...")
audio_target = librosa.resample(audio_full, orig_sr=sr_orig, target_sr=TARGET_SR)

# ── Extract clips ──────────────────────────────────────────────────────────
PAD_SECONDS = 0.3  # padding before/after each segment to avoid mid-word cuts

def extract_clips(segments, label, audio, sr):
    # Filter by minimum length, sort by duration descending
    usable = [(s, e, e - s) for s, e in segments if (e - s) >= MIN_SEGMENT_SECONDS]
    usable.sort(key=lambda x: x[2], reverse=True)
    selected = usable[:MAX_CLIPS_PER_SPEAKER]

    print(f"\n{label}: extracting {len(selected)} clips (from {len(usable)} usable segments)")

    for i, (start, end, dur) in enumerate(selected, 1):
        # Apply time offset (SRT starts at 01:00:00) and add padding
        real_start = start - OFFSET - PAD_SECONDS
        real_end = end - OFFSET + PAD_SECONDS

        start_sample = int(real_start * sr)
        end_sample = int(real_end * sr)

        # Bounds check
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        clip = audio[start_sample:end_sample]

        # Normalize to -1dB peak
        peak = np.max(np.abs(clip))
        if peak > 0:
            clip = clip / peak * 0.89  # ~-1dB

        fname = f"{label}_{i:02d}_{dur:.1f}s.wav"
        out_path = OUT_DIR / fname
        sf.write(str(out_path), clip, sr)
        print(f"  {fname}  (src {real_start:.1f}s - {real_end:.1f}s)")


extract_clips(male_segs, "MALE", audio_target, TARGET_SR)
extract_clips(female_segs, "FEMALE", audio_target, TARGET_SR)

# ── Summary ────────────────────────────────────────────────────────────────
all_clips = list(OUT_DIR.glob("*.wav"))
print(f"\n{'='*60}")
print(f"Extracts saved to: {OUT_DIR}")
print(f"Total files: {len(all_clips)}")
for f in sorted(all_clips):
    info = sf.info(str(f))
    print(f"  {f.name}  ({info.duration:.1f}s, {info.samplerate}Hz)")
print(f"\nNext steps:")
print(f"  1. Listen to clips, delete any with overlap/noise")
print(f"  2. Pick best clip per speaker (15-20s clean solo speech)")
print(f"  3. Move finalized clips to voices/ parent directory")
print(f"  4. Use as audio_prompt_path in Chatterbox TTS")
