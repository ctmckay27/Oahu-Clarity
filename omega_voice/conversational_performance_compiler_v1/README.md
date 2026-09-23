# MARI_CONVERSATIONAL_PERFORMANCE_COMPILER_v1

Executable middle layer between Mari's conversational state and the verified
one-session native voice renderer.

## Why this exists

The repaired native continuity route can preserve Mari's selected G1-1 speaker
identity and deliver complete intelligible utterances, but prior categorical
thought-to-prosody control sounded acted rather than conversational. The later
raw human performance-reference experiment was rejected because the donor
acoustic prefix strongly changed speaker identity.

CPC v1 therefore uses a stricter pipeline:

```
conversation state
  -> communicative action
  -> discourse / thought groups
  -> information focus
  -> turn controller
  -> typed microtiming
  -> respiration intent
  -> continuous performance coordinates
  -> qualified native projection
  -> one-session Mari renderer
```

## Implemented systems

### 1. Communicative-state resolver
Resolves what the utterance is doing: sharing, clarifying, repairing,
challenging, inviting a response, setting a boundary, admitting, etc. The state
is derived from explicit conversational structure and supplied intent/feedback,
not from a hidden-emotion classifier.

### 2. Thought/discourse groups
Preserves conversational-v6 source units and adds explicit relations such as
`REVISES_PREVIOUS`, `CONTRASTS_WITH_PREVIOUS`, and `INVITES_RESPONSE`.

### 3. Information-focus model
Tracks given/new/contrastive lexical structure. Focus is represented separately
from actuation so "important" does not automatically become "speak louder".

### 4. Continuous performance space
Builds continuous utterance-progress trajectories for:
- finality / continuation
- tempo tendency
- energy tendency
- articulation tendency
- pitch-movement tendency
- vocal-weight tendency
- focus strength

Only finality/continuation currently leaves the representation because it has an
existing measured same-speaker native direction. The other coordinates remain
explicit and unapplied until independently qualified.

### 5. Microtiming engine
Keeps distinct events distinct:
- thought boundary
- retrieval hesitation
- self-correction
- emphasis preparation
- turn hold / yield
- physical intake request

None of these events can reintroduce source-token withholding.

### 6. Respiration actuator boundary
`MARI_NATIVE_RESPIRATION_ACTUATOR_v1` is implemented as a fail-closed actuator
contract. Physical intake requests are preserved, but the current native
renderer has no qualified non-starving intake mechanism. Source withholding,
waveform stitching, fake breath clips, decoder restarts, and speaker
substitution are forbidden. A future candidate must pass identity, continuity,
source-consumption, audible-consequence, and intelligibility gates before it can
be considered for promotion.

### 7. Turn controller
Represents entry latency, resume/new-turn state, floor hold/yield, and expected
listener response. Entry latency is interaction scheduling outside the waveform,
not synthesized silence.

### 8. Native acoustic adapter
Projects discourse-derived boundary finality into the existing measured
same-speaker native trajectory with a smooth local envelope while preserving:
- selected Mari Voice v1 / G1-1 x-vector profile
- one decoder session
- one model invocation
- zero unit restarts
- zero source-release holds
- no waveform stitching
- no generic TTS fallback
- no raw cross-speaker performance donor

### 9. Ordinary-interaction learning interface
Natural reactions such as "too acted", "wrong stress", "timing is weird", or
"that doesn't sound like Mari" are captured as scoped evidence and routed to the
responsible component. Numeric ratings are not required and the system does not
silently self-modify from one reaction.

## Evidence boundary

CPC v1 is a real executable compiler and native adapter, not a claim that Mari's
performance is finished. It can demonstrate context-sensitive acoustic
consequences while preserving current continuity constraints. Perceptual
naturalness still requires listening evidence.

The physical respiration actuator also remains unresolved at the mechanism
level. The interface is built; false execution claims are prohibited.
