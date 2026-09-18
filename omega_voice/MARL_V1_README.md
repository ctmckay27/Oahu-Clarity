# Mari Acoustic Realization Layer v1.0

Status: **ENGINEERING COMPLETE / ACOUSTIC IDENTITY OPEN**.

This is the executable coherence layer between Mari structural vocal identity and replaceable acoustic renderers. It owns renderer-search controls, experiment lineage, qualitative listening feedback, current admission state, anchor promotion, and verification. It does not own Mari semantic identity.

## Current source bindings

- Voice resolution: `MARI_VOICE_RESOLUTION_STATE_OBJECT_v1.5_2026-09-17`
- Generation gate: `MARI_VOICE_GENERATION_GATE_v3.1_2026-09-17`
- Continuous carrier: `MARI_CONTINUOUS_SPEECH_CARRIER_ARCHITECTURE_v1.1_2026-09-17`
- Voice basis: `MARI_VOICE_BASIS_STATE_OBJECT_v1.0_2026-09-17`
- Voice machine predecessor: `MARI_VOICE_MACHINE_STATE_v0.3_2026-09-18`

## What is closed

- typed persistent state;
- Qwen renderer calibration and reference-clone receipts;
- controlled search-plan compiler;
- qualitative/directional Carl feedback interface;
- vague-rejection non-learning rule;
- explicit calibration-vs-promotable experiment modes;
- fail-closed v3.1 promotion gate;
- Carl-only anchor promotion;
- separate engineering readiness and Mari-voice readiness;
- exact state persistence/readback route;
- round-one renderer execution.

Successor tests: **23/23 passed**.

## What is deliberately open

`voice_v1_ready = false`. There is no accepted Mari acoustic anchor. Current v3.1 also keeps the promotion route closed until its basis/project-owned-generator conditions are satisfied or that governing gate is explicitly superseded. Qwen round-one outputs are therefore renderer-calibration probes, not canon candidates.

## Current round

`MARL-R1-A`, `MARL-R1-B`, and `MARL-R1-C` are attached under `rounds/MARL-R1/`. Carl can respond qualitatively (for example: “B is closer; darker; less polished; keep the timing”) and the layer moves only explicitly named dimensions. Numeric ratings are optional, not required.