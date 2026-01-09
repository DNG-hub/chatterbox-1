# Ghost Voice Filters - Safe Reconnection Guide

This guide explains how to safely reconnect the Voice Filters to `gradio_tts_turbo_app.py` without breaking the TTS generation system.

## ⚠️ Critical Principle

**Voice Filters are POST-PROCESSING only** - They should NEVER modify the `generate()` function or interfere with TTS generation. Filters are applied AFTER audio is generated, as a separate step.

## 📋 Pre-Reconnection Checklist

- [ ] TTS generation is working correctly (test with simple text)
- [ ] Temperature set to 0.8 (original working value)
- [ ] All dependencies installed: `pip install librosa soundfile numpy scipy pedalboard`
- [ ] You have a backup or can restore from git if needed

## 🔄 Reconnection Methods

### Method 1: Incremental Reconnection (Recommended - Smoke Test One at a Time)

This is the safest approach. Test each component separately before adding the next.

#### Step 1: Enable Basic Filter (Smoke Test)

1. Open `gradio_tts_turbo_app.py`
2. Change line 16: `VOICE_FILTERS_ENABLED = False` → `VOICE_FILTERS_ENABLED = True`
3. Save and restart the app
4. **Test**: Generate audio - TTS should still work exactly the same (filters aren't connected to UI yet)

#### Step 2: Add Simple Filter UI (Minimal Integration)

Add this code **after** `audio_output = gr.Audio(label="Output Audio")` (around line 206):

```python
            # Ghost Filter - Simple Test
            if GHOST_FILTERS_AVAILABLE:
                with gr.Accordion("🎭 Ghost Voice Filter (Test)", open=False):
                    ghost_mode = gr.Dropdown(
                        choices=["None"] + get_mode_names(),
                        value="None",
                        label="Filter Mode"
                    )
                    apply_ghost_filter_btn = gr.Button("Apply Ghost Filter", variant="secondary")
                    filtered_audio_output = gr.Audio(label="Filtered Output")
```

#### Step 3: Connect Simple Button Handler

Add this **after** the main `run_btn.click()` handler (around line 234):

```python
    # Ghost filter - Simple test connection
    if GHOST_FILTERS_AVAILABLE:
        def simple_apply_filter(audio_tuple, mode):
            if audio_tuple is None or mode == "None":
                return None
            return apply_ghost_filter_to_gradio_audio(audio_tuple, mode)
        
        apply_ghost_filter_btn.click(
            fn=simple_apply_filter,
            inputs=[audio_output, ghost_mode],
            outputs=filtered_audio_output,
        )
```

4. **Test**: 
   - Generate audio normally (should work)
   - Apply a filter to the generated audio (should work)
   - Verify original audio is unchanged

#### Step 4: Add Full Parameterized UI (Optional - Advanced)

Only after Step 3 works, you can add the full parameterized UI. See the commented section at the end of the file for the full UI code.

---

### Method 2: Full Reconnection at Once

If you're confident, you can reconnect everything at once:

1. Set `VOICE_FILTERS_ENABLED = True` (line 16)
2. Uncomment and add the full UI section (see end of file)
3. Uncomment the full button handler (see end of file)

**⚠️ Risk**: If something breaks, it's harder to identify which component caused it.

---

## 🧪 Smoke Testing Strategy

### Test 1: Basic TTS (Before Reconnection)
```
Test text: "Hello world, this is a test."
Expected: Clean audio output, words match exactly
```

### Test 2: Filter Import (After Step 1)
```
Check console: Should see no errors about Ghost filters
Expected: App loads without errors
```

### Test 3: Simple Filter (After Step 3)
```
1. Generate: "Hello world"
2. Select mode: "whisper_chorus"
3. Click "Apply Ghost Filter"
Expected: Filtered audio plays, original unchanged
```

### Test 4: Each Mode (After Full Reconnection)
Test each mode individually:
- `whisper_chorus` - Should sound layered/ethereal
- `spore_cloud` - Should sound granular/particle-like
- `mycelium_pulse` - Should sound deep/underground
- `resonance_capture` - Should sound glitched/corrupted
- `transmission` - Should sound clean/broadcast

### Test 5: Parameterized Controls (After Step 4)
```
1. Apply filter with default parameters
2. Adjust one parameter (e.g., reverb_room_size)
3. Re-apply filter
Expected: Audio changes based on parameter
```

---

## 🚨 Warning Signs to Watch For

If you see any of these, **STOP** and revert:

1. **TTS words don't match input** - Filters are interfering with generation (shouldn't happen if filters are post-processing only)
2. **Audio is silent or corrupted** - Filter processing error
3. **App crashes on startup** - Import error or syntax issue
4. **Temperature seems wrong** - Something is modifying generation parameters
5. **Generation is slower** - Filters might be running during generation (wrong place)

---

## 🔧 Troubleshooting

### Issue: "Ghost filters not available" warning
**Solution**: Check that `src/Voice Filters/ghost_filters.py` exists and dependencies are installed

### Issue: Filtered audio is silent
**Solution**: Check audio format conversion in `apply_ghost_filter_to_gradio_audio()`

### Issue: TTS quality degraded after reconnection
**Solution**: Verify filters are NOT being applied during generation - they should only run when you click "Apply Ghost Filter"

### Issue: App won't start
**Solution**: 
1. Set `VOICE_FILTERS_ENABLED = False` to disable
2. Check for syntax errors in the UI code you added
3. Verify all imports are correct

---

## 📝 Key Points

1. **Filters are POST-PROCESSING ONLY** - Never modify `generate()` function
2. **Original audio is preserved** - Filters create a separate output
3. **Test incrementally** - Add one piece at a time, test, then add the next
4. **Keep original working** - If filters break, you can always disable them

---

## 🔗 Related Documentation

- `PARAMETER_GUIDE.md` - Full parameter reference
- `README.md` - Basic usage and integration examples
- `ghost_filters.py` - Core filter implementation
- `ghost_filters_parameterized.py` - Advanced parameterized version

---

## ✅ Reconnection Checklist

When reconnecting, verify:

- [ ] TTS generation works identically before and after
- [ ] Original audio output is unchanged
- [ ] Filtered audio is a separate output
- [ ] All 5 modes work
- [ ] No errors in console
- [ ] Audio plays correctly
- [ ] Parameters adjust the sound as expected

---

## 🎯 Quick Reconnection (Minimal)

If you just want the basics working:

1. Set `VOICE_FILTERS_ENABLED = True`
2. Add simple UI (Step 2 above)
3. Add simple handler (Step 3 above)
4. Test with one mode
5. Done!

The full parameterized UI can be added later if needed.
