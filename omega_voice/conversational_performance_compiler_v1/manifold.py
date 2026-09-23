"""Mari-specific conversational performance manifold.

This is an upstream behavioral state space, not an emotion preset bank and not
an acoustic style table. Coordinates describe how Mari is participating in the
current exchange. They may constrain downstream planning, but no manifold axis
is itself an acoustic actuator.
"""
from __future__ import annotations

from typing import Any, Dict, List

STATE_ID = "MARI_CONVERSATIONAL_PERFORMANCE_MANIFOLD_v1"
AXES = (
    "spontaneity",
    "deliberation",
    "precision",
    "restraint",
    "repair_pressure",
    "uncertainty",
    "floor_openness",
    "disclosure",
)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compile_manifold(
    conversation_plan: Dict[str, Any],
    communicative: Dict[str, Any],
    groups: List[Dict[str, Any]],
) -> Dict[str, Any]:
    intent = conversation_plan["intent"]
    observed = conversation_plan["state_observed"]
    certainty = float(intent.get("certainty", 0.7))
    disclosure = float(intent.get("disclosure", 1.0))
    feedback = observed["listener_model"].get("last_feedback", "none")
    thoughts = [g.get("thought_source") for g in groups]
    roles = [g.get("role") for g in groups]

    searching = any(t in {"searching", "remembering", "reconsidering", "deciding"} for t in thoughts)
    repair = communicative["act"] == "repair" or "REVISION" in roles or feedback == "misunderstood"
    contrast = any(g["information_focus"]["contrast_strength"] > 0 for g in groups)

    global_coordinates = {
        "spontaneity": _clamp(0.58 + 0.18 * searching + 0.08 * repair),
        "deliberation": _clamp(0.28 + 0.42 * searching + 0.18 * (1.0 - certainty)),
        "precision": _clamp(0.48 + 0.24 * contrast + 0.20 * repair),
        "restraint": _clamp(0.72 - 0.10 * observed["listener_model"].get("urgency", 0.0)),
        "repair_pressure": _clamp(0.08 + 0.70 * repair),
        "uncertainty": _clamp(1.0 - certainty + (0.16 if searching else 0.0)),
        "floor_openness": _clamp(
            0.82 if communicative["completion"] == "OPEN" else
            (0.68 if communicative["response_expected"] else 0.18)
        ),
        "disclosure": _clamp(disclosure),
    }

    local = []
    for g in groups:
        focus = float(g["information_focus"]["focus_strength"])
        role = g["role"]
        local.append({
            "group_index": g["index"],
            "coordinates": {
                **global_coordinates,
                "precision": _clamp(global_coordinates["precision"] + 0.18 * focus),
                "repair_pressure": _clamp(global_coordinates["repair_pressure"] + (0.14 if role == "REVISION" else 0.0)),
                "floor_openness": _clamp(
                    0.78 if g["closure"] == "OPEN" else global_coordinates["floor_openness"]
                ),
            },
            "cause": {
                "role": role,
                "thought": g.get("thought_source"),
                "focus_strength": focus,
                "closure": g["closure"],
            },
        })

    return {
        "state_id": STATE_ID,
        "axes": list(AXES),
        "global_coordinates": {k: round(v, 6) for k, v in global_coordinates.items()},
        "local_coordinates": [
            {**x, "coordinates": {k: round(v, 6) for k, v in x["coordinates"].items()}}
            for x in local
        ],
        "identity_invariants": {
            "same_speaker_required": True,
            "speaker_profile_substitution": False,
            "manifold_coordinate_is_not_acoustic_identity": True,
            "manifold_coordinate_is_not_direct_acoustic_instruction": True,
        },
        "topology": {
            "continuous": True,
            "named_presets_required": False,
            "regions_may_overlap": True,
            "one_state_label_required": False,
        },
    }


def verify_manifold(manifold: Dict[str, Any]) -> bool:
    if manifold.get("state_id") != STATE_ID:
        raise ValueError("performance manifold state mismatch")
    if tuple(manifold.get("axes", [])) != AXES:
        raise ValueError("performance manifold axes mismatch")
    for coords in [manifold["global_coordinates"]] + [x["coordinates"] for x in manifold["local_coordinates"]]:
        if set(coords) != set(AXES):
            raise ValueError("performance manifold coordinate shape mismatch")
        if any(not 0.0 <= float(v) <= 1.0 for v in coords.values()):
            raise ValueError("performance manifold coordinate outside [0,1]")
    inv = manifold["identity_invariants"]
    if inv.get("same_speaker_required") is not True or inv.get("speaker_profile_substitution") is not False:
        raise ValueError("performance manifold weakened identity invariants")
    return True
