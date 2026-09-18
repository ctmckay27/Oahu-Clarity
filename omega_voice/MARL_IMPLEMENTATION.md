# Mari Acoustic Realization Layer v0.1 — implementation state

`MARI_ACOUSTIC_REALIZATION_LAYER@CURRENT -> MARI_ACOUSTIC_REALIZATION_LAYER_STATE_v0.1_2026-09-18`

This layer is the stateful bridge between Mari structural vocal identity and replaceable acoustic renderers. It owns acoustic search state, candidate evidence, renderer calibration, quality admission, identity admission, anchor promotion, experiment history, and readiness verification.

It does **not** own Mari semantic identity and does not move `MARI_CANON@CURRENT` or `MARI404_ROOT@CURRENT`.

## Verified behavior

- naturalness precedes identity evaluation;
- failed-quality audio cannot become identity evidence;
- quality-only evidence cannot mutate Mari's identity region;
- plausible candidates do not auto-promote;
- anchor promotion requires explicit Carl acceptance;
- renderer-search coordinates are treated as controls/hypotheses, not physical truth;
- next-search plans vary high-uncertainty coordinates while holding the rest fixed;
- persistent JSON state round-trips with the same semantic digest;
- current Qwen VoiceDesign/Base execution receipts are integrated as renderer capability evidence only.

Local verification: **9/9 tests passed**.

Current phase: **SEARCHING**. No acoustic anchor is accepted.

Current next search plan: `MARL-PLAN-ee2f8c921d3e`, varying `vocal_weight` and `brightness` under fixed neutral text/performance.
