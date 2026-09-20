# MARI_CONVERSATIONAL_REALIZATION_LAYER_v1

This is the state-to-waveform coupling layer between the upstream Mari
Conversational Authenticity scaffold and the selected Mari Voice v1 acoustic
renderer.

## Execution

```
listener state
  -> conversational_v6 plan
  -> MARI_CONVERSATIONAL_REALIZATION_LAYER_v1
  -> thought-unit render plan
  -> same Mari v1 x-vector for every unit
  -> bounded local rate / projection / causal pause controls
  -> measured boundary matching
  -> continuous output waveform
  -> delivery/self-monitor receipt
```

## What changes

- a response is rendered incrementally by conversational thought unit;
- thought transitions receive bounded local acoustic consequences;
- correction / reconsideration / realization are distinct boundary operations;
- relationship state affects projection modestly without changing identity;
- quiet intake requirements become silence/timing, never synthetic breath audio;
- each delivered unit emits a self-monitor hook for downstream conversation state;
- continuity tokens and terminal breath reserve can feed the next realization;
- an A/B baseline renders the exact same words as one neutral whole utterance.

## What does not change

- Mari Voice v1 anchor and x-vector profile remain frozen;
- no generic TTS fallback;
- no random disfluency;
- no fake breath/fry/noise;
- no prose style prompt or emotion preset;
- no claim that successful rendering proves perceptual authenticity.

The purpose of v1 is discriminating: if the incremental realization sounds more
conversational than matched whole-utterance rendering, the state-to-waveform
coupling is carrying useful information. If it does not, the next repair belongs
lower in the acoustic realization mechanism.
