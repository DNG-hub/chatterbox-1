# Session Handoff - 2026-03-15

## Completed This Session
- Extracted male and female reference voice clips from NotebookLM podcast audio
- Source: `E:\DaVinci_Projects\Cat_Daniel_Collapse_Protocol\Youtube\Podcast\Collapse Protocol Prolog\VO\Why_the_world_is_collapsing_on_purpose.m4a`
- User annotated `PrologPodcast.srt` with `[Male]`/`[Female]` alternating tags and additional `[male]...[/male]` bracketed tags for extra male segments
- Created `extract_reference_voices.py` — SRT-driven voice extraction script with pyannote-free approach
- Extracted 10 raw clips (5 male, 5 female) to `E:\DaVinci_Projects\Cat_Daniel_Collapse_Protocol\storyteller\doc\voices\raw_extracts\`
- Finalized best reference voices to `E:\REPOS\StoryTeller\docs\voices\`:
  - `Male.wav` (18.4s, 24kHz) — longest clean male segment
  - `Female.wav` (25.2s, 24kHz) — longest clean female segment

## In Progress
- No blockers. Reference voices are ready for use with Chatterbox TTS.

## Next Steps
1. Test reference voices with Chatterbox TTS/Turbo to validate quality
2. Clean up `raw_extracts/` directory once finalized voices are confirmed good
3. If voice quality needs improvement, user can annotate `TransPod.srt` (21-min podcast) for additional material

## Context Notes
- Chatterbox uses 15s conditioning window (`ENC_COND_LEN = 15 * S3_SR`) — clips above 15s are ideal
- Minimum reference audio is >5s (hard assertion in code)
- Splicing clips together is NOT recommended — introduces discontinuities in voice embedding
- Minor truncation at clip boundaries (mid-word cuts from SRT timing) does not significantly impact voice encoder quality
- 300ms padding was added to segment boundaries to reduce truncation
- pyannote diarization pipeline is gated — user would need to accept terms at HuggingFace if ever needed
- The podcast voices are NOT Cat & Daniel — they are unnamed male/female NotebookLM voices

## Technical State
- Branch: master
- Last commit: 6574910 - "feat: add PTT press click at start of voice_comm filter"
- Uncommitted: yes — Ghost filter changes, extract_reference_voices.py, various reconstruction scripts

## Quick Resume
Start next session with: `/startup` or "Read .claude/handoff.md and continue where we left off"
