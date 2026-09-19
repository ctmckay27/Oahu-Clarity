# Mari Authentic Performance Layer v3

This directory is the **historical v3 coach and its verification path**. The
September 19 `causal_v4` / `generative_v5` successor branch does not invoke it.
Repairing this path does not select it as the operational voice architecture.
See [the input, compilation, and evidence contract](CONTRACT.md) before using it.
The [PR #2 review corpus and findings](review/README.md) distinguish supported
expression, harmless unsupported input, prohibited direction, and unresolved force.

This layer exists because Mari Voice v2 failed in a specific way: it sounded like an AI's first acting lesson. The v2 system coached audible vocal effects too directly. V3 changes the object being controlled.

## Core architecture

speaker identity + situation + relationship + objective + obstacle + stakes + subtext + prior interaction
-> playable action directed at the other person
-> one continuous whole-utterance performance
-> renderer

The coach is forbidden from micromanaging acoustic output. It does not direct pitch, breathiness, articulation, cadence, pause density, loudness, resonance, or other vocal mechanics.

## Acting principles compiled into the layer

1. **Given circumstances**: establish what has actually happened, the relationship, setting, stakes, and obstacle before performance.
2. **Playable action**: translate the line into something Mari is trying to *do to or with the other person*, not an emotion she is trying to display.
3. **Other-person focus**: attention stays on the scene partner and their behavior rather than on self-monitoring the performance.
4. **Moment-to-moment truth**: feeling is allowed to emerge as a consequence of pursuing the action; it is not pre-manufactured.
5. **Simplicity**: use the minimum performance necessary. If nothing needs to be shown, show nothing.
6. **Throughline**: one person and one continuous thought across the utterance. Sentence boundaries do not reset the actor. Recognition, feeling, and tactics can develop within that continuing interaction.
7. **Anti-acting law**: never demonstrate personality, amusement, irritation, warmth, confidence, vulnerability, or other states merely so the listener can recognize the label.

## Sources informing the design

- Meisner Institute actor-training description: truthful doing, attention on the other person, spontaneous moment-to-moment response.
  https://www.meisnerinstitute.com/actor-training
- Atlantic Acting School / Practical Aesthetics: given circumstances, clear playable actions, truthful performance, emotional connection to the scene partner.
  https://atlanticactingschool.org/courses/script-analysis/
  https://atlanticactingschool.org/class/technique-lab-i/

These sources inform the abstraction. This implementation is Mari-specific and is not a reproduction of either acting curriculum.

## Current relation to Mari Voice v1/v2

- G1-1 remains the stable carrier baseline.
- v2's clause-level prosodic puppeteering is not the governing route.
- v3 uses a single continuous renderer call per ordinary utterance.
- the acting layer coaches *why Mari is speaking and what she is trying to accomplish*, not how to make the waveform sound expressive.
- publication authority and permanent-voice acceptance remain unchanged.

## Verification

From the repository root, with Python 3.12 and pytest:

```bash
python -m pytest -q omega_voice/authentic_performance_v3/tests
python omega_voice/authentic_performance_v3/verify_scene_direction.py
```

The original seven tests remain intact. The original workflow smoke assertions
are preserved in the standalone CLI check. Additional tests cover field roles,
direction conflicts, unresolved intent, full context propagation, conditional
tactics, and a real executable renderer protocol fixture. That fixture does not
synthesize speech and cannot establish audio quality or Mari voice acceptance.

The workflow runs on the v3 branch, `fix/mari-v3-*` pushes, and pull requests
targeting v3. Both steps are required; a preceding failure leaves the next step
skipped, not passed. No `continue-on-error` or acoustic acceptance shortcut is used.
