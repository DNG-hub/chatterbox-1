# Ghost Voice Filters for Chatterbox

Voice processing filters for Cat & Daniel: Collapse Protocol's Ghost character.

## Available Modes

| Mode | Description |
|------|-------------|
| `whisper_chorus` | Multiple minds speaking as one - layered, ethereal |
| `spore_cloud` | Voice condensing from particles - granular, drifting |
| `mycelium_pulse` | Underground network voice - deep, rhythmic, organic |
| `resonance_capture` | Corrupted recordings - glitched, static, absorbed voices |
| `transmission` | Direct audience address - clean broadcast quality |

## Dependencies

```bash
pip install librosa soundfile numpy scipy pedalboard
```

## Integration with Chatterbox Gradio

### Option 1: Quick Integration

Add to your Chatterbox `app.py` after audio generation:

```python
import sys
sys.path.append("src/Voice Filters")
from ghost_filters import apply_ghost_filter, GHOST_MODES

# In your Gradio interface, add:
ghost_mode = gr.Dropdown(
    choices=list(GHOST_MODES.keys()),
    value="whisper_chorus",
    label="Ghost Voice Filter"
)

apply_filter_btn = gr.Button("Apply Ghost Filter")

# Wire the button
def apply_filter(audio, mode):
    if audio is None:
        return None
    sr, data = audio
    # Convert to float if needed
    if data.dtype != np.float32:
        data = data.astype(np.float32) / 32768.0
    processed = apply_ghost_filter(data, sr, mode)
    # Convert back to int16 for Gradio
    processed_int = (processed * 32767).astype(np.int16)
    return (sr, processed_int)

apply_filter_btn.click(
    fn=apply_filter,
    inputs=[audio_output, ghost_mode],
    outputs=[audio_output]
)
```

### Option 2: Standalone Processing

```python
from ghost_filters import process_file

# Process a WAV file
process_file(
    input_path="output.wav",
    output_path="output_ghost.wav",
    mode="whisper_chorus"
)
```

### Option 3: Direct Array Processing

```python
from ghost_filters import apply_ghost_filter
import librosa

# Load audio
y, sr = librosa.load("input.wav", sr=None)

# Apply filter
processed = apply_ghost_filter(y, sr, mode="resonance_capture")

# Save
import soundfile as sf
sf.write("output.wav", processed, sr)
```

## Mode Selection Guide

| Context | Recommended Mode |
|---------|------------------|
| Speaking to Cat alone | `whisper_chorus` |
| Physical manifestation (spores visible) | `spore_cloud` |
| Long-distance / underground | `mycelium_pulse` |
| Through terminals / old recordings | `resonance_capture` |
| Prologue / epilogue / audience address | `transmission` |
