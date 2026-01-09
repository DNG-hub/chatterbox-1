# Ghost Filter Parameters Guide

Complete reference for all adjustable parameters in the Ghost Voice Filters.

## 🎚️ Common Effects (All Modes)

### Static/Noise Parameters
Control the background static/noise layer that simulates unconventional media:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Static Level** | 0.0 - 0.15 | Mode-specific | Volume of background static (0 = none, higher = more static) |

**Mode-Specific Static Types:**
- **Whisper Chorus**: High-frequency tape hiss (8-12 kHz) - Default: 0.02 (Low)
- **Spore Cloud**: Mid-range crackle/granular (2-5 kHz) - Default: 0.025 (Low-Medium)
- **Mycelium Pulse**: Low-frequency rumble (50-200 Hz) - Default: 0.03 (Medium)
- **Resonance Capture**: Broad-spectrum white noise - Default: 0.03 (Medium)
- **Transmission**: Radio static/interference (1-4 kHz) - Default: 0.02 (Low)

**Tips:**
- **Low (0.01-0.02)**: Subtle background presence, barely noticeable
- **Medium (0.025-0.04)**: Clearly audible, adds authenticity
- **High (0.05-0.08)**: Prominent static, may obscure voice details
- **Very High (0.1-0.15)**: Heavy static, voice may be difficult to understand

**Note:** Static is always present at the set level (constant, not reactive to voice). Each mode has a characteristic static type that matches its tonal quality.

### Reverb Parameters
Control the spatial/echo characteristics:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Reverb Room Size** | 0.0 - 1.0 | Mode-specific | Larger = more spacious, cavernous sound |
| **Reverb Damping** | 0.0 - 1.0 | Mode-specific | Higher = less echo, tighter reverb tail |
| **Reverb Wet Level** | 0.0 - 1.0 | Mode-specific | Amount of reverb effect (0 = dry, 1 = fully wet) |
| **Reverb Dry Level** | 0.0 - 1.0 | Mode-specific | Amount of original signal (0 = no original, 1 = full original) |

**Tips:**
- Increase `reverb_wet` + decrease `reverb_dry` for more ethereal/spacious sound
- Higher `room_size` = more "underground cavern" feel
- Lower `damping` = longer, more echoey reverb tail

### Chorus Parameters
Add modulation/doubling effects:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Chorus Rate (Hz)** | 0.0 - 2.0 | Mode-specific | Speed of modulation (0.1-0.5 = slow, 1.0+ = fast) |
| **Chorus Depth** | 0.0 - 1.0 | Mode-specific | Amount of pitch variation |
| **Chorus Mix** | 0.0 - 1.0 | Mode-specific | Blend of chorus effect (0 = none, 1 = full) |

**Tips:**
- Lower rate (0.1-0.3) = subtle, slow modulation
- Higher rate (0.8-1.5) = faster, more noticeable warble
- Increase depth for more dramatic effect

### Delay Parameters
Add echo/repeat effects:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Delay Time (s)** | 0.0 - 0.5 | Mode-specific | Echo delay in seconds |
| **Delay Feedback** | 0.0 - 1.0 | Mode-specific | How much echo repeats (0 = single echo, 1 = many repeats) |
| **Delay Mix** | 0.0 - 1.0 | Mode-specific | Echo volume relative to original |

**Tips:**
- Short delay (0.05-0.15s) = subtle doubling
- Long delay (0.2-0.5s) = distinct echo
- High feedback = cascading echoes

## 🎛️ EQ & Frequency

### Frequency Filters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Highpass Cutoff (Hz)** | 20 - 2000 | Mode-specific | Removes frequencies below (higher = removes more bass) |
| **Lowpass Cutoff (Hz)** | 1000 - 20000 | Mode-specific | Removes frequencies above (lower = more muffled/distant) |

**Tips:**
- Highpass at 200-500 Hz = removes rumble, makes voice "float"
- Lowpass at 3000-6000 Hz = telephone/radio effect
- Combine both for "tunnel" or "distant" sound

### Compression

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Compressor Threshold (dB)** | -40 - 0 | Mode-specific | When compression kicks in (lower = more compression) |
| **Compressor Ratio** | 1 - 10 | Mode-specific | Compression strength (higher = more squashed) |

**Tips:**
- Lower threshold (-30 to -20) = more consistent volume
- Higher ratio (5-10) = more "punchy" or "squashed" sound
- Use for broadcast/radio effect

### Gain

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Gain (dB)** | -10 - 10 | Mode-specific | Overall volume boost/cut |

**Tips:**
- Use to compensate for filter volume changes
- +2 to +5 dB = subtle boost
- -3 to -5 dB = quieter, more subtle

## 🎨 Mode-Specific Parameters

### Whisper Chorus

Controls the mix of multiple voice layers:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Original Layer Mix** | 0.0 - 1.0 | 0.55 | Main voice volume |
| **Low Pitch Layer Mix** | 0.0 - 1.0 | 0.20 | -1 semitone layer |
| **High Pitch Layer Mix** | 0.0 - 1.0 | 0.15 | +0.5 semitone layer |
| **Undertone Layer Mix** | 0.0 - 1.0 | 0.10 | -3 semitone whisper layer |

**Tips:**
- Increase undertone mix for more "whisper" effect
- Increase high pitch for brighter, more ethereal
- Balance all layers for "multiple minds" effect

### Spore Cloud

Controls granular/particle effects:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Grain Rate (Hz)** | 1 - 20 | 8 | Particle modulation speed |
| **Pulse Frequency (Hz)** | 0.1 - 2.0 | 0.5 | Slow breathing pulse |
| **Time Stretch Rate** | 0.95 - 1.1 | 1.02 | Warping amount |

**Tips:**
- Higher grain rate (12-20) = faster, more chaotic particles
- Lower grain rate (3-6) = slower, more organic
- Lower time stretch (0.98-1.0) = less warping, more natural
- Higher time stretch (1.05-1.1) = more warped, otherworldly

### Mycelium Pulse

Controls underground/network effects:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Pulse Rate (Hz)** | 0.1 - 2.0 | 0.7 | Rhythmic pulse speed |
| **Drone Frequency (Hz)** | 20 - 60 | 35 | Subsonic drone pitch |
| **Drone Amplitude** | 0.0 - 0.2 | 0.08 | Drone volume |
| **Pitch Shift (semitones)** | -5 - 5 | -2 | Overall pitch adjustment |

**Tips:**
- Lower pulse rate (0.3-0.5) = slower, more organic pulse
- Higher pulse rate (1.0-1.5) = faster, more urgent
- Increase drone amplitude for more "underground rumble"
- Lower pitch shift (-3 to -4) = deeper, more menacing
- Higher pitch shift (0 to +2) = less deep, more accessible

### Resonance Capture

Controls glitch/corruption effects:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Static Level** | 0.0 - 0.15 | 0.03 | Broad-spectrum white noise amount (see Static/Noise section above) |
| **Bitcrush Depth (bits)** | 4 - 16 | 10 | Distortion amount (lower = more distorted) |
| **Echo Mix** | 0.0 - 0.3 | 0.08 | Absorbed voice echo volume |

**Tips:**
- Static level (0.03 default) = constant background noise, characteristic of corrupted media
- Lower bitcrush (4-6 bits) = heavy distortion, lo-fi
- Higher bitcrush (12-16 bits) = subtle degradation
- Increase echo mix for more "absorbed voices" effect

### Transmission

Controls broadcast/clean effects:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Pitch Shift (semitones)** | -5 - 5 | -0.5 | Subtle pitch adjustment |

**Tips:**
- Keep pitch shift small (-1 to +1) for broadcast quality
- Larger shifts change character more dramatically

## 🎛️ Overall Control

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Filter Intensity** | 0.0 - 2.0 | 1.0 | Overall filter strength multiplier |
| **Static Level Override** | 0.0 - 0.15 | None (uses mode default) | Override mode's default static level |

**Tips:**
- **Filter Intensity:**
  - 0.0 = No filter (original audio)
  - 0.5 = Subtle effect
  - 1.0 = Default preset strength
  - 1.5 = Stronger effect
  - 2.0 = Extreme effect (may be too much)
- **Static Level Override:**
  - Leave as None to use mode's characteristic static level
  - Set to 0.0 to disable static completely
  - Adjust to fine-tune static volume independently from other effects

## 🎯 Quick Experimentation Guide

### For More "Spooky"
- Increase reverb wet level (0.4-0.6)
- Increase chorus depth (0.3-0.5)
- Lower lowpass cutoff (4000-6000 Hz)
- Increase filter intensity (1.2-1.5)

### For More "Mechanical"
- Increase bitcrush (lower bit depth: 6-8)
- Increase static level (0.05-0.08)
- Add delay with feedback (0.3-0.5)
- Lower highpass cutoff (100-150 Hz) for more bass

### For More "Ethereal"
- Increase reverb room size (0.6-0.8)
- Increase chorus mix (0.3-0.5)
- Higher lowpass cutoff (10000-15000 Hz)
- Increase delay mix (0.2-0.4)

### For More "Underground/Deep"
- Lower pitch shift (-3 to -4 semitones)
- Increase drone amplitude (0.12-0.18)
- Lower pulse rate (0.4-0.6 Hz)
- Lower highpass cutoff (40-80 Hz)

## 💡 Pro Tips

1. **Start with defaults**: Use mode presets first, then tweak
2. **One parameter at a time**: Change one thing, listen, adjust
3. **Use intensity slider**: Quick way to scale entire effect
4. **Combine parameters**: Reverb + Chorus + Delay = rich effects
5. **EQ is powerful**: Highpass/Lowpass can dramatically change character
6. **Save your settings**: Note parameter values that work well

## 🔧 Technical Notes

- Parameters set to `None` use mode defaults
- All parameters are mode-aware (only relevant ones apply)
- Audio is automatically normalized to prevent clipping
- Processing happens in real-time (may take a few seconds for long audio)
