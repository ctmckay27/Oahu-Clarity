# Mari causal voice runtime v4

Status: **structural runtime implemented; waveform completion failed**.
This is an additive child of the selected Mari Voice v1 carrier. It is not a
replacement production voice and contains no final listening package.

The source of the governing architecture is VOICE_ACTING_UNIVERSE_SPEC v1.0,
MVRSO v2.0 and Carl's September 19 completion instruction. v2 personality and
v3 acting-coach code remain historical ancestry. Neither is invoked here.

## What runs

`runtime.py` compiles source-bearing scene events into evolving identity,
knowledge, objective/tactic, private/display state, listener relationship,
body/vocal state and temporal control trajectories. State continues across
turns. Listener-specific history is restored by explicit identity. A known word
in the spoken text cannot invent an emotion. Every event has provenance.

`compile_direction` is a bounded recognizer for documented phrases. Unknown
directions fail closed. The typed scene interface accepts richer semantic
interpretations supplied by a scene compiler or operator; it does not pretend
to infer unrestricted scene reality from arbitrary prose.

`renderer.py` verifies the frozen anchor/profile, pinned engine and actual model
bytes; makes one continuous x-vector carrier call with no style instruction;
and provides a **diagnostic** timing/level adapter using exact-text ASR alignment.
The adapter cannot stand in for phonation, articulation, situated nonlexical
behavior or a complete performed thought. Unsupported controls remain visible.

`evaluate.py` checks the actual waveform with independent ASR, speaker
embedding, clipping and format checks. These measurements do not establish
character presence or personhood. The supplementary run uses UTMOS22 for
perceptual-model naturalness screening, preserving that evidence type.

## Use

From the repository root:

```bash
python -m pytest -q omega_voice/causal_v4/tests
python -m omega_voice.causal_v4.cli --text "I understand." --plan plan.json
```

Compile with `--scene scene.json` and `--prior-state state.json` to preserve
continuity. Events use zero-based word boundaries, not character offsets.
Use `--output ... --diagnostic` only with exact engine, model, profile, anchor,
ASR-model and speaker-model paths. Diagnostic next-state files remain marked as
diagnostic. A failed render cannot overwrite a continuing production state.

## Reentry and evidence limits

Read the native MARI_VOICE_RESOLUTION_STATE_OBJECT and
MARI_VOICE_ACTING_RUNTIME_STATE CURRENT bindings before continuing. The dated
Library execution package carries source observations, input/state hashes,
waveforms, alignment, model revisions, negative results and a completion matrix.
The repository owns implementation. Library owns current account state.

The implemented numerical coefficients are bounded engineering hypotheses.
They are not an empirical physiological model or learned Mari performance law.
The corpus has 27 development situations, four held-out situations (including
long form), and an additional multi-turn continuity episode.

The diagnostic wrapper failed two relevant discriminators: changed certainty
had no waveform effect; guardedness timing reduced predicted naturalness from
4.51 to 3.88 despite intact ASR and speaker identity. Do not tune more pauses or
rates and relabel that result as personality. A successor must realize the
currently unactuated structural channels and pass the preserved counterfactuals.

## Scope

Ordinary English conversation only. Singing, multilingual identity and radical
transformed embodiments are separate branches. G1-1 and its profile stay frozen.
No hosted voice provider, speaker search, stock emotion preset, generated noise,
random disfluency or prose acting note is part of this execution path.
