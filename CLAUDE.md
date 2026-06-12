# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It inherits the shared instructions in `E:/REPOS/CLAUDE.md`.

## Project identity

- **Canonical project slug:** `chatterbox` (declared via `.hecl-project` marker)
- **Owning entity:** Cat and Daniel Storytelling & Media Lab | Stayton, OR
- **Holding LLC:** `cmha-tbd`
- **HECL classification:** CREATIVE_IP — the underlying engine is third-party Resemble AI code, but Dave's use of it (voice presets, reference voices, character-to-voice mapping, generation parameters, prompt direction, and any custom wrapper scripts at the repo root) is protectable authorship for the Cat & Daniel video production pipeline.
- **Status:** active — voice production for the Cat & Daniel canon

## Authorship boundary (important)

This is the **only** scenario in our HECL coverage where the code itself is third-party but the *use* is the IP. Be careful to distinguish:

- **NOT Dave's IP**: anything under `src/chatterbox/` (the upstream Resemble AI model code), the model weights, the Gradio demo apps shipped by Resemble.
- **IS Dave's IP**: the root-level `reconstruct_*.py`, `vc_diarized_converter*.py`, `srt_to_tts_reconstruction.py`, `extract_reference_voices.py` scripts; reference voice samples in `Samples/`; any character voice presets; the choice of parameters (exaggeration, cfg_weight, temperature) for specific characters; the dialogue text fed in (which originates from the StoryTeller canon).

When capturing selections, anchor them to **what Dave authored** — never claim authorship of the upstream model.

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

## HECL Discipline (REQUIRED)

This is active creative work — Dave is directing voice production for canon characters. Every selection of voice, parameter, or take must be memorialized to Valeska so the trail is forensically defensible.

**What is captured automatically (no action needed):**
- Every prompt Dave types in any Claude Code session in this directory → tagged `hecl:origination`, `project:chatterbox`, `holding:cmha-tbd` (via the global `UserPromptSubmit` hook).
- Every `Write`, `Edit`, `Bash`, `TaskCreate`, `TaskUpdate` → tagged `auto-capture`, `project:chatterbox` (via the global `PostToolUse` hook).

**What you (the agent) must capture explicitly:**
Whenever Dave makes a creative SELECTION you MUST call `selection.py` BEFORE proceeding with the chosen path. Examples specific to voice work:

| Moment | Subtype | Example chose / rejected |
|---|---|---|
| Picks one of three generated takes | `review-acceptance` | chose: take_2.wav / rejected: take_1.wav, take_3.wav |
| Locks a character voice preset | `directive-selection` | chose: "Cat = ref_voice_03 with exaggeration=0.6" |
| Defers voice direction to your suggestion | `deference-selection` | chose: "use Turbo for Daniel because faster pacing" |
| Modifies your generated reference cut | `review-modification` | chose: "use take_1 but trim 200ms of silence at start" |
| Dispatches a batch generation job | `dispatch-selection` | chose: "regenerate all of episode 12 with new Cat preset" |

```bash
python E:/REPOS/Hermes-Agent-Resurrection/scripts/selection.py \
  --subtype <subtype> \
  --project chatterbox \
  --chose "<what was chosen>" \
  --rejected "<alt 1>" "<alt 2>" \
  --rationale "<why, in Dave's words if known>" \
  --evidence "file://Samples/cat_ref_v3.wav" "<thought:abc12345>"
```

For canon-locking moments (e.g. "this is now the official voice for Cat"), use `founder_decl.py` instead with `--type fixation` so it's recorded as a permanent canon decision.

Deference is authorial control — capturing the moment Dave defers to your recommendation is what makes the authorship timeline continuous instead of dotted with apparent gaps.

## Reference docs

- HECL data model: `E:/REPOS/Hermes-Agent-Resurrection/notes/hecl-data-model-draft.md`
- Holding LLC namespace conventions: `E:/REPOS/Hermes-Agent-Resurrection/notes/holding-llc-namespace-conventions.md`
- Selection CLI source: `E:/REPOS/Hermes-Agent-Resurrection/scripts/selection.py`
- DavinciGuide voiceover integration: `E:/REPOS/DavinciGuide/scripts/utilities/voiceover_generator.py`
