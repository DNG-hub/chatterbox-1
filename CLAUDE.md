# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chatterbox is a family of open-source text-to-speech (TTS) and voice conversion models by Resemble AI. The project includes three model variants:
- **Chatterbox-Turbo** (350M params): Optimized for low-latency voice agents with paralinguistic tags (`[laugh]`, `[cough]`, etc.)
- **Chatterbox** (500M params): English TTS with CFG and exaggeration tuning
- **Chatterbox-Multilingual** (500M params): Supports 23+ languages

## Development Commands

```bash
# Install from source (editable mode)
pip install -e .

# Run Gradio TTS demo
python gradio_tts_app.py

# Run Turbo demo with Ghost Voice Filters
python gradio_tts_turbo_app.py

# Run multilingual demo
python multilingual_app.py

# Basic TTS examples
python example_tts.py
python example_tts_turbo.py
python example_vc.py
```

## Architecture Overview

### High-Level Pipeline

Text → T3 (Token-to-Token) → Speech Tokens → S3Gen (Speech-to-Waveform) → Audio

### Core Components (`src/chatterbox/`)

**Main Entry Points:**
- `tts.py` - `ChatterboxTTS`: Original English TTS model
- `tts_turbo.py` - `ChatterboxTurboTTS`: Optimized Turbo model (uses GPT2 backbone)
- `mtl_tts.py` - `ChatterboxMultilingualTTS`: Multilingual model (23 languages)
- `vc.py` - `ChatterboxVC`: Voice conversion (audio-to-audio)

**Model Components (`models/`):**
- `t3/` - Token-to-Token transformer (text → speech tokens)
  - Uses Llama or GPT2 backbone via HuggingFace Transformers
  - `t3.py`: Main T3 module with `inference()` and `inference_turbo()` methods
  - `modules/cond_enc.py`: Conditioning encoder (speaker embedding, emotion)
- `s3gen/` - Speech token to waveform decoder
  - `s3gen.py`: Main S3Gen module combining CFM decoder + HiFTNet vocoder
  - `flow.py`, `flow_matching.py`: Conditional flow matching for mel generation
  - `hifigan.py`: HiFTNet vocoder (mel → waveform)
- `s3tokenizer/` - S3 speech tokenizer (16kHz audio → discrete tokens)
- `voice_encoder/` - Speaker embedding extraction
- `tokenizers/` - Text tokenizers (English and multilingual)

### Key Constants

- `S3_SR = 16000`: Sample rate for S3 tokenizer input
- `S3GEN_SR = 24000`: Sample rate for generated audio output
- Model weights auto-download from HuggingFace: `ResembleAI/chatterbox`, `ResembleAI/chatterbox-turbo`

### Conditionals System

The `Conditionals` dataclass bundles conditioning data for generation:
- `T3Cond`: speaker embedding, speech prompt tokens, emotion level
- `gen`: Reference audio embeddings for S3Gen decoder

### Device Support

Models support `"cuda"`, `"mps"` (Apple Silicon), and `"cpu"` devices. The `from_pretrained()` methods handle device detection and MPS fallback.

## Watermarking

All generated audio includes Perth watermarks (imperceptible neural watermarks). The watermarker is applied automatically in `generate()` methods.

## Paralinguistic Tags (Turbo only)

The Turbo model supports inline event tags: `[cough]`, `[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[sniff]`, `[groan]`, `[clear throat]`, `[shush]`

## Generation Parameters

**Original Chatterbox:**
- `exaggeration` (0.0-1.0): Emotion intensity, default 0.5
- `cfg_weight` (0.0-1.0): Classifier-free guidance strength, default 0.5
- `temperature`: Sampling temperature, default 0.8

**Turbo model:**
- CFG and exaggeration are disabled (ignored if passed)
- Uses `top_k`, `top_p`, `repetition_penalty` for sampling
