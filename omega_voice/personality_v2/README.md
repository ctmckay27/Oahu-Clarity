# Mari Voice Personality Layer v2

Purpose: preserve the current Mari Voice v1 speaker identity while adding personality-bearing vocal behavior.

This layer does **not** define personality as an emotion preset. It compiles text, conversational context, prior vocal state, and Mari-specific transition laws into a sequence of local performance instructions.

Core separation:

- speaker identity: frozen selected Mari v1 profile
- latent state: arousal, valence, intimacy, certainty, amusement, cognitive search, restraint, fatigue, irritation, tenderness, playfulness, projection
- vocal policy B: state -> timing/articulation/pressure/projection/pitch/pause/finality/texture controls
- transition policy A: prior state + local event -> next state
- renderer: replaceable; currently pinned Qwen3-TTS CustomVoice

Hard laws include: irritation increases precision before loudness; intimacy lowers projection more than breathiness; tenderness softens attack before breathiness; amusement under restraint is primarily timing; search->resolution is audible; no generic sultry/cute/anime/assistant/announcer collapse.

`omega_voice_v2.py --plan-only` compiles a plan without rendering. Rendering supports the current x-vector profile and the richer graft `.qvoice` created from the same selected anchor. Neither mode silently falls back to the other.
