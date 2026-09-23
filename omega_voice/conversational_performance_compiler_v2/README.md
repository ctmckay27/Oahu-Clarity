# MARI_CONVERSATIONAL_PERFORMANCE_COMPILER_v2

CPC v2 is an additive successor to CPC v1.

It preserves the v1 conversational/discourse compiler and measured same-speaker
finality/continuation actuation, and adds exactly one newly qualified native
dimension: speaking tempo.

Tempo was derived from same-Mari fast/slow calibration references and admitted
only after held-out free generation preserved intelligibility and Mari speaker
similarity while producing the expected duration ordering.

Not promoted:
- onset pressure: rejected for inconsistent held-out effect and identity risk;
- local focus: rejected because the held-out effect reversed direction;
- respiration/recovery: still unqualified.

The runtime therefore uses a two-axis native bank: finality/continuation plus
tempo. No acting presets, raw cross-speaker references, waveform stitching,
source withholding, fake breaths, decoder restarts, or generic TTS fallback.
