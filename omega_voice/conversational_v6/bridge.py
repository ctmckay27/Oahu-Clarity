"""Bridge Mari conversational plans into the existing causal voice runtime."""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from .runtime import verify_plan


def compile_causal(
    conversational_plan: Dict[str, Any],
    *,
    voice_prior: Optional[Dict[str, Any]] = None,
    temporal_policy: str = "listener_causal_v6",
) -> Dict[str, Any]:
    verify_plan(conversational_plan)
    if temporal_policy not in {"listener_causal_v6", "contrast_focus_v7"}:
        raise ValueError("conversational bridge requires listener-aware causal policy")
    from omega_voice.causal_v4.runtime import compile_scene

    scene = {
        "events": copy.deepcopy(conversational_plan["causal_events"]),
        "metadata": {
            "conversational_plan_hash": conversational_plan["plan_hash"],
            "renderer_contract": copy.deepcopy(conversational_plan["renderer_contract"]),
            "turn_taking": copy.deepcopy(conversational_plan["turn_taking"]),
            "units": [
                {
                    "index": u["index"],
                    "start_word": u["start_word"],
                    "end_word": u["end_word"],
                    "thought": u["thought"],
                    "speech_act": u["speech_act"],
                    "quiet_intake_before_s": u["quiet_intake_before_s"],
                }
                for u in conversational_plan["units"]
            ],
        },
    }
    return compile_scene(
        conversational_plan["response_text"],
        scene,
        prior=voice_prior,
        policy=temporal_policy,
    )
