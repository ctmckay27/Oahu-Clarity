# Mari Authentic Performance Layer v3

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
6. **Throughline**: one person and one continuous thought across the utterance. Sentence boundaries do not reset the actor.
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
