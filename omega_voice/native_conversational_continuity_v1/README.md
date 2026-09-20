# MARI_NATIVE_CONVERSATIONAL_CONTINUITY_LAYER_v1

This layer replaces thought-unit waveform stitching with a single continuous
native Qwen generation request.

## Execution path

```text
listener turn
  -> MARI_CONVERSATIONAL_AUTHENTICITY_STATE plan
  -> source-bound text-token release schedule
  -> measured native boundary-contour trajectory
  -> one Mari Voice v1 x-vector
  -> one decoder / KV-cache / codec stream
  -> one continuous waveform
  -> verified consumption and continuity receipt
```

## What is native and continuous

- one model invocation for the full utterance;
- one decoder session;
- one persistent KV cache and codec history;
- no per-unit seeds or synthesis restarts;
- no waveform concatenation or crossfade repair;
- conversational text becomes available incrementally inside the same request;
- a measured same-speaker finality/continuation direction changes locally by
  conversational state while the decoder remains live.

## Evidence boundary

The only acoustic direction used in v1 comes from
`MARI_ISOLATED_NATIVE_CONTROL_EVIDENCE_2026-09-19.zip`. The repository uses an
integrity-checked rank-two compact projection with cosine similarity
`0.9797614813` to the measured float32 direction and relative L2 error
`0.2001689970`. The original evidence archive remains authoritative.

This is a boundary-contour actuator, not an emotion, personality or generic
style vector. Unsupported acoustic dimensions remain unresolved. The layer does
not add random filler, breath, fry, noise, emotion presets, prose acting
instructions, or a fallback voice.

## Verification target

The production workflow creates a matched three-way comparison:

- **A:** one neutral whole-utterance Mari v1 render;
- **B:** the predecessor stitched incremental realization;
- **C:** this layer, one native continuous request with internal state changes.

A successful workflow proves execution, identity locking, text-release
consumption and one-session continuity. Carl's listening comparison determines
whether C is perceptually more person-like.
