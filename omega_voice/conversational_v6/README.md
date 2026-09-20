# Mari Conversational Authenticity Scaffold v1

This package adds an upstream conversational-performance layer without replacing Mari Voice v1 acoustic identity or the existing causal runtime.

## Why it exists

A coherent carrier is not enough to sound like a person genuinely talking to a listener. The missing structure is modeled before acoustic rendering:

1. persistent conversational mind,
2. listener-specific state,
3. thought-before-language state,
4. incremental speech units,
5. self-monitoring and repair,
6. relationship-conditioned behavior,
7. embodied phrase and reserve budgeting,
8. causal microbehavior only,
9. turn-taking and response latency,
10. compilation into the existing time-evolving causal voice runtime.

## Non-negotiable laws

- Acoustic identity is frozen to the selected Mari Voice v1 anchor/profile at this layer.
- Lexical text alone does not infer hidden emotion or listener state.
- No random filler, fake breaths, generic "humanization," style tags, or generic TTS fallback.
- Planned speech is not conversational history until it is actually delivered.
- If delivery is interrupted, unheard units remain intention, not committed speech.
- Relationship and listener observations are source-grounded inputs, not guessed personality labels.
- This scaffold improves causal structure; it does not itself prove perceptual authenticity.

## Main API

```python
from omega_voice.conversational_v6 import (
    new_conversation_state,
    plan_turn,
    compile_causal,
    commit_delivery,
)

state = new_conversation_state(listener_id="carl")
plan = plan_turn(
    state,
    "That still sounds too performed.",
    "Yeah. The carrier is coherent now, but the thought process is still too prepackaged.",
    observation={"feedback": "misunderstood", "relationship": {"familiarity": .95}},
    intent={"tactic": "clarify", "thought": "known"},
)
causal_plan = compile_causal(plan)
state = commit_delivery(state, plan)
```

`compile_causal` uses the existing listener-aware causal policy and does not send prose acting directions to the renderer.
