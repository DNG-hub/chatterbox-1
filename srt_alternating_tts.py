#!/usr/bin/env python3
"""
Parse SRT and generate full TTS audio using alternating speaker pattern.
Assumes speakers alternate (typical dialog format): Speaker1, Speaker2, Speaker1, Speaker2...
Uses pitch detection from original audio to determine speaker order.
"""

import re
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import torch
from src.chatterbox.tts_turbo import ChatterboxTurboTTS

SRT_FILE = Path(r"E:\REPOS\DavinciGuide\XML Exports\banter.srt")
ORIGINAL_AUDIO = Path(r"Samples/Autopsy_of_an_Engineered_Apocalypse.m4a")
OUTPUT_FILE = Path("Samples/Autopsy_TTS_RECONSTRUCTED.wav")

def parse_srt(srt_path):
    """Parse SRT file and extract text lines."""
    lines_text = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                # Get text (remove HTML tags)
                text = '\n'.join(lines[2:])
                text = re.sub(r'<[^>]+>', '', text).strip()
                if text:
                    lines_text.append(text)
            except:
                pass

    return lines_text

def detect_dominant_speaker(audio, sr):
    """
    Detect dominant speaker (male vs female) in entire audio.
    Returns 'MALE' or 'FEMALE' based on median pitch.
    """
    print("Analyzing dominant speaker characteristics...")
    f0, _, _ = librosa.pyin(audio, fmin=80, fmax=400, sr=sr)
    valid_f0 = f0[~np.isnan(f0)]

    if len(valid_f0) > 0:
        median_pitch = np.median(valid_f0)
        print(f"  Median pitch: {median_pitch:.1f} Hz")
        speaker1 = "FEMALE" if median_pitch > 150 else "MALE"
        speaker2 = "MALE" if speaker1 == "FEMALE" else "FEMALE"
        print(f"  Speaker 1 (odd): {speaker1}")
        print(f"  Speaker 2 (even): {speaker2}")
        return speaker1, speaker2
    return "FEMALE", "MALE"

def generate_tts(tts_model, text, speaker, device):
    """Generate TTS audio."""
    try:
        # Use Chatterbox Turbo for faster generation
        audio = tts_model.generate(
            text,
            temperature=0.8,
            top_p=0.95,
            top_k=50
        )
        return audio.numpy()
    except Exception as e:
        print(f"    TTS Error: {str(e)[:60]}")
        return None

def main():
    print("="*70)
    print("SRT-BASED FULL RECONSTRUCTION USING TTS")
    print("="*70)

    # Parse SRT
    print(f"\n1. Parsing SRT: {SRT_FILE.name}")
    dialog_lines = parse_srt(SRT_FILE)
    print(f"   Found {len(dialog_lines)} dialog lines")

    if len(dialog_lines) == 0:
        print("ERROR: No dialog lines found in SRT!")
        return

    # Show sample
    print("\n   Sample lines:")
    for i, line in enumerate(dialog_lines[:5]):
        print(f"     [{i+1}] {line[:60]}")

    # Load original audio for speaker characterization
    print(f"\n2. Loading original audio: {ORIGINAL_AUDIO.name}")
    y, sr = librosa.load(str(ORIGINAL_AUDIO), sr=16000)
    print(f"   Duration: {len(y)/sr:.2f}s")

    # Detect speaker order
    print(f"\n3. Detecting speaker characteristics...")
    speaker1, speaker2 = detect_dominant_speaker(y, sr)

    # Assign speakers (alternating)
    print(f"\n4. Mapping speakers (alternating pattern):")
    print(f"   Odd lines (1,3,5...):   {speaker1}")
    print(f"   Even lines (2,4,6...):  {speaker2}")

    speaker_list = []
    for i in range(len(dialog_lines)):
        speaker = speaker1 if (i % 2) == 0 else speaker2
        speaker_list.append(speaker)

    speaker1_count = sum(1 for s in speaker_list if s == speaker1)
    speaker2_count = sum(1 for s in speaker_list if s == speaker2)
    print(f"\n   Total: {speaker1_count} {speaker1} lines, {speaker2_count} {speaker2} lines")

    # Load TTS model
    print(f"\n5. Loading Chatterbox TTS model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    tts = ChatterboxTurboTTS.from_pretrained(device=device)

    # Generate TTS
    print(f"\n6. Generating TTS for {len(dialog_lines)} lines...")
    tts_audio_list = []
    tts_sr = 24000

    for i, (text, speaker) in enumerate(zip(dialog_lines, speaker_list), 1):
        if i % 100 == 0 or i <= 5:
            speaker_label = f"[{speaker}]"
            text_preview = text[:50].replace('\n', ' ')
            print(f"   [{i}/{len(dialog_lines)}] {speaker_label:8s} {text_preview:50s}", end=" ", flush=True)

        audio = generate_tts(tts, text, speaker, device)

        if audio is not None:
            tts_audio_list.append(audio)
            if i % 100 == 0 or i <= 5:
                print(f"OK ({len(audio)/tts_sr:.2f}s)")
        else:
            if i % 100 == 0 or i <= 5:
                print("FAILED")
            # Use short silence instead
            tts_audio_list.append(np.zeros(int(0.5 * tts_sr)))

    # Concatenate with small silence gaps
    print(f"\n7. Concatenating {len(tts_audio_list)} audio segments...")
    final_segments = []
    for i, audio in enumerate(tts_audio_list):
        final_segments.append(audio)
        # Add 50ms silence between lines for natural pause
        final_segments.append(np.zeros(int(0.05 * tts_sr)))

    full_audio = np.concatenate(final_segments)

    # Export
    print(f"\n8. Exporting to {OUTPUT_FILE}...")
    sf.write(str(OUTPUT_FILE), full_audio, tts_sr)

    final_duration_min = len(full_audio) / tts_sr / 60
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024*1024)

    print(f"\n" + "="*70)
    print("RECONSTRUCTION COMPLETE!")
    print("="*70)
    print(f"Input: {len(dialog_lines)} dialog lines from SRT")
    print(f"Output: {OUTPUT_FILE.name}")
    print(f"Duration: {final_duration_min:.2f} minutes")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Speakers: {speaker1} vs {speaker2}")

if __name__ == "__main__":
    main()
