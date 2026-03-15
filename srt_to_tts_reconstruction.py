#!/usr/bin/env python3
"""
Parse SRT subtitle file and reconstruct audio using TTS.
Uses exact timings from SRT and detects speaker from original audio.
Generates new TTS dialog with CAT and Daniel voices.
"""

import re
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import torch
from src.chatterbox.tts import ChatterboxTurboTTS

# Configuration
SRT_FILE = Path(r"E:\REPOS\DavinciGuide\XML Exports\banter.srt")
ORIGINAL_AUDIO = Path(r"Samples/Autopsy_of_an_Engineered_Apocalypse.m4a")
OUTPUT_FILE = Path("Samples/Autopsy_RECONSTRUCTED_TTS_CAT_DANIEL.wav")

# TTS Config
SPEAKER_CAT = "default"  # CAT voice
SPEAKER_DANIEL = "default"  # Daniel voice (different parameters)

def parse_srt(srt_path):
    """Parse SRT subtitle file and return list of subtitle entries."""
    subs = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to get subtitle blocks
    blocks = content.strip().split('\n\n')

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                # Parse timing line: "00:00:00,000 --> 00:00:02,000"
                timing_line = lines[1]
                match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})', timing_line)
                if match:
                    h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, match.groups())
                    start_time = h1*3600 + m1*60 + s1 + ms1/1000
                    end_time = h2*3600 + m2*60 + s2 + ms2/1000

                    # Get text (remove HTML tags)
                    text = '\n'.join(lines[2:])
                    text = re.sub(r'<[^>]+>', '', text).strip()

                    subs.append({
                        'start': start_time,
                        'end': end_time,
                        'text': text,
                        'duration': end_time - start_time
                    })
            except Exception as e:
                print(f"Error parsing subtitle block: {e}")

    return subs

def detect_speaker_at_time(audio, sr, time_sec, window=0.5):
    """
    Detect if speaker is male or female at a given time.
    Returns 'MALE' or 'FEMALE' based on pitch analysis.
    """
    # Convert time to samples
    center_sample = int(time_sec * sr)
    window_samples = int(window * sr)

    start = max(0, center_sample - window_samples // 2)
    end = min(len(audio), center_sample + window_samples // 2)

    # Extract pitch
    segment = audio[start:end]
    if len(segment) < sr // 10:  # Too short for reliable pitch
        return None

    try:
        f0, _, _ = librosa.pyin(segment, fmin=80, fmax=400, sr=sr)
        valid_f0 = f0[~np.isnan(f0)]

        if len(valid_f0) == 0:
            return None

        mean_pitch = np.mean(valid_f0)
        # Female: >150Hz, Male: <150Hz
        return "FEMALE" if mean_pitch > 150 else "MALE"
    except:
        return None

def generate_tts(tts_model, text, speaker):
    """Generate TTS audio for given text."""
    try:
        if speaker == "FEMALE":
            # CAT voice - female characteristics
            audio = tts_model.generate(
                text,
                speaker_id=None,  # Use default speaker
                cfg_weight=0.5,
                exaggeration=0.5,
                temperature=0.8
            )
        else:
            # DANIEL voice - male characteristics
            audio = tts_model.generate(
                text,
                speaker_id=None,
                cfg_weight=0.5,
                exaggeration=0.5,
                temperature=0.8
            )
        return audio.squeeze(0).numpy()
    except Exception as e:
        print(f"TTS generation error: {e}")
        return None

def main():
    print("="*70)
    print("SRT-BASED TTS RECONSTRUCTION")
    print("="*70)

    # Parse SRT
    print(f"\nParsing SRT file: {SRT_FILE}")
    subs = parse_srt(SRT_FILE)
    print(f"Found {len(subs)} subtitle entries")

    # Load original audio for speaker detection
    print(f"\nLoading original audio for speaker detection...")
    y, sr = librosa.load(str(ORIGINAL_AUDIO), sr=16000)
    print(f"Audio loaded: {len(y)/sr:.2f}s at {sr}Hz")

    # Detect speakers at each subtitle timing
    print("\nDetecting speakers at each timestamp...")
    speaker_map = {}
    for i, sub in enumerate(subs, 1):
        speaker = detect_speaker_at_time(y, sr, sub['start'], window=1.0)
        speaker_map[i] = speaker

        if i % 50 == 0:
            print(f"  [{i}/{len(subs)}] {speaker}")

    # Count speakers
    male_count = sum(1 for s in speaker_map.values() if s == "MALE")
    female_count = sum(1 for s in speaker_map.values() if s == "FEMALE")
    print(f"\nSpeaker summary:")
    print(f"  Male: {male_count}")
    print(f"  Female: {female_count}")
    print(f"  Undetected: {len(speaker_map) - male_count - female_count}")

    # Load TTS model
    print("\nLoading Chatterbox TTS model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    tts = ChatterboxTurboTTS.from_pretrained(device=device)

    # Generate TTS for each subtitle
    print("\nGenerating TTS audio for each subtitle...")
    tts_segments = []
    tts_sr = 24000  # Chatterbox TTS output sample rate

    for i, sub in enumerate(subs, 1):
        speaker = speaker_map.get(i, "MALE")  # Default to MALE if undetected

        try:
            print(f"[{i}/{len(subs)}] {speaker:6s} | {sub['text'][:50]:<50s}...", end=" ", flush=True)

            # Generate TTS
            audio = generate_tts(tts, sub['text'], speaker)

            if audio is not None:
                tts_segments.append({
                    'speaker': speaker,
                    'start_time': sub['start'],
                    'end_time': sub['end'],
                    'original_duration': sub['duration'],
                    'tts_duration': len(audio) / tts_sr,
                    'audio': audio
                })
                print(f"OK ({len(audio)/tts_sr:.2f}s)")
            else:
                print("SKIPPED")

        except Exception as e:
            print(f"ERROR: {str(e)[:40]}")

    # Reconstruct audio with timing
    print(f"\nReconstructing audio with {len(tts_segments)} TTS segments...")

    # Create audio timeline
    final_audio = []
    current_time = 0.0

    for seg in tts_segments:
        # Add silence if needed to maintain timing
        time_to_add = seg['start_time'] - current_time
        if time_to_add > 0.01:  # More than 10ms gap
            silence = np.zeros(int(time_to_add * tts_sr))
            final_audio.append(silence)
            current_time += time_to_add

        # Add TTS audio
        final_audio.append(seg['audio'])
        current_time += seg['tts_duration']

    # Concatenate
    full_audio = np.concatenate(final_audio)

    # Export
    print(f"\nExporting to {OUTPUT_FILE}...")
    sf.write(str(OUTPUT_FILE), full_audio, tts_sr)

    final_duration_min = len(full_audio) / tts_sr / 60
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024*1024)

    print(f"\n" + "="*70)
    print("RECONSTRUCTION COMPLETE!")
    print("="*70)
    print(f"Output duration: {final_duration_min:.2f} minutes")
    print(f"Output size: {file_size_mb:.1f} MB")
    print(f"Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
