# Ghost Voice Filter - Realistic Static Design

This document describes the real-world phenomena that inspire each Ghost mode's static pattern and how they are implemented.

---

## Real-World Static Phenomena

### Atmospheric/Lightning Noise (Sferics)
- **Character**: Impulse-based crackles and pops from lightning discharges
- **Behavior**: Random timing (Poisson distribution), damped oscillation decay
- **Frequency**: Broadband, more energy at lower frequencies
- **Variation**: Nearby storms = loud pops; distant = cumulative noise floor

### Multipath Fading
- **Character**: Signal strength varies as radio waves take multiple paths
- **Behavior**: Rayleigh distribution for random fading, slow drift over time
- **Effect**: Creates organic "breathing" in signal level

### Tape Hiss & Degradation
- **Character**: High-frequency noise from magnetic particle randomness
- **Behavior**: Relatively steady but with micro-variations
- **Artifacts**: Dropouts (oxide shedding), wow/flutter (speed variation)

### Vinyl Crackle
- **Character**: Physical medium damage creates impulse events
- **Behavior**: Random pops from dust/scratches, density varies
- **Pattern**: Can have rhythmic component (rotation period)

### Radio/EVP Artifacts
- **Character**: Interference, sweep fragments, AM modulation
- **Behavior**: Bursts of interference, frequency-shifted artifacts
- **Effect**: Suggests "other voices" or transmissions bleeding through

---

## Mode-Specific Static Design

### 1. Whisper Chorus
**Ambiance**: Intimate recording of multiple consciousnesses, like voices captured on aged tape in a quiet room.

**Static Concept**: Vintage tape recorder aesthetic
- **Base**: High-frequency tape hiss (8-12 kHz)
- **Level**: Low and intimate (0.015-0.025)
- **Fading**: Very subtle drift (±15%), slow (0.3 Hz)
- **Impulses**: Rare, soft micro-pops (1-2 per second)
- **Dropouts**: Occasional brief level dips (tape wear)

**Emotional Effect**: Creates sense of fragile, captured moment

---

### 2. Spore Cloud
**Ambiance**: Voice forming from particles, condensing from mist, organic and alive.

**Static Concept**: Granular, particle-like texture
- **Base**: Mid-frequency crackle (2-6 kHz)
- **Level**: Low-medium (0.02-0.035)
- **Fading**: Breathing pattern (±25%), medium speed (0.5 Hz)
- **Impulses**: Granular bursts - clusters of tiny particles (8-15 per second, very quiet)
- **Dropouts**: None (continuous organic texture)

**Emotional Effect**: Living, breathing, emergent presence

---

### 3. Mycelium Pulse
**Ambiance**: Ancient network beneath the earth, deep rumbling communication through soil and roots.

**Static Concept**: Underground rumble with earthen movement
- **Base**: Sub-bass rumble (30-150 Hz)
- **Level**: Medium (0.025-0.04)
- **Fading**: Slow organic drift (±30%), very slow (0.15 Hz)
- **Impulses**: Deep thuds/shifts - like earth settling (0.5-1 per second)
- **Dropouts**: Slow "pressure" variations (like being underground)

**Emotional Effect**: Primordial, vast, patient

---

### 4. Resonance Capture
**Ambiance**: Corrupted recording device absorbing voices, haunted technology breaking down.

**Static Concept**: Degraded electronics, EVP characteristics
- **Base**: Broadband white noise with instability (100-8000 Hz)
- **Level**: Medium-high (0.03-0.05)
- **Fading**: Rayleigh fading - erratic signal (±40%), variable speed (0.3-1.5 Hz)
- **Impulses**: Heavy - pops, clicks, digital glitches (5-12 per second)
- **Dropouts**: Frequent signal breaks (corrupted medium)
- **Special**: Occasional "sweep" artifacts (radio bleed-through feel)

**Emotional Effect**: Unstable, haunted, reality breaking down

---

### 5. Transmission
**Ambiance**: Direct broadcast from another realm, breaking through interference to reach the audience.

**Static Concept**: Radio broadcast fighting through atmospheric interference
- **Base**: Radio static band (1-5 kHz)
- **Level**: Low (0.015-0.025)
- **Fading**: Multipath fading pattern (±20%), medium (0.4 Hz)
- **Impulses**: Interference bursts - sharp but infrequent (2-4 per second)
- **Dropouts**: Brief signal fades (atmospheric conditions)
- **Special**: AM modulation wobble on the static itself

**Emotional Effect**: Distant but intentional communication, breaking through

---

## Implementation Parameters

### Core Static Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `base_level` | Center point for static amplitude | 0.01 - 0.08 |
| `level_variance` | How much level drifts (%) | 0.0 - 0.5 |
| `fade_rate` | Speed of level drift (Hz) | 0.1 - 2.0 |
| `fade_complexity` | Number of overlapping fade waves | 1 - 4 |

### Impulse Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `impulse_density` | Average impulses per second | 0 - 20 |
| `impulse_variance` | Randomness in timing | 0.0 - 1.0 |
| `impulse_amplitude` | Relative loudness of pops | 0.5 - 3.0 |
| `impulse_decay` | How quickly impulses fade (ms) | 1 - 50 |
| `impulse_clustering` | Tendency to cluster together | 0.0 - 1.0 |

### Dropout Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `dropout_probability` | Chance per second | 0.0 - 2.0 |
| `dropout_duration` | Length of dropout (ms) | 10 - 200 |
| `dropout_depth` | How much signal drops | 0.3 - 1.0 |

### Frequency Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `freq_center` | Center frequency of noise band | 50 - 10000 Hz |
| `freq_width` | Bandwidth of noise | 50 - 5000 Hz |
| `freq_drift` | Random drift in filter (%) | 0.0 - 0.2 |

---

## Mode Preset Summary

| Mode | Base Type | Level | Fading | Impulses | Dropouts | Character |
|------|-----------|-------|--------|----------|----------|-----------|
| Whisper Chorus | Tape hiss | 0.02 | Subtle | Rare/soft | Occasional | Intimate, aged |
| Spore Cloud | Granular | 0.025 | Breathing | Particle clusters | None | Organic, alive |
| Mycelium Pulse | Rumble | 0.03 | Slow drift | Deep thuds | Pressure waves | Ancient, deep |
| Resonance Capture | Unstable | 0.04 | Erratic | Heavy glitches | Frequent | Corrupted, haunted |
| Transmission | Radio | 0.02 | Multipath | Interference | Signal fades | Broadcast, distant |
