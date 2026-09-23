"""Microtiming intent compiler.

Timing phenomena remain typed so syntax, cognition, repair, turn exchange, and
respiration are not collapsed into one generic "pause". Internal timing intent
never becomes source-token withholding in this layer.
"""
from __future__ import annotations

from typing import Any, Dict, List


_EVENT_DEFAULTS = {
    "SYNTACTIC_BOUNDARY": 0.08,
    "THOUGHT_BOUNDARY": 0.12,
    "RETRIEVAL_HESITATION": 0.16,
    "SELF_CORRECTION": 0.10,
    "EMPHASIS_PREPARATION": 0.04,
    "TURN_YIELD": 0.0,
    "TURN_HOLD": 0.0,
    "PHYSICAL_INTAKE_REQUEST": 0.0,
}


def compile_microtiming(plan: Dict[str, Any], groups: List[Dict[str, Any]], turn: Dict[str, Any]) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    units_by_index = {u["index"]: u for u in plan["units"]}

    for group in groups:
        unit = units_by_index[group["unit_index"]]
        micro = set(group.get("explicit_microbehavior", []))
        if group["index"] > 0:
            events.append({
                "kind": "THOUGHT_BOUNDARY",
                "group_index": group["index"],
                "requested_duration_s": _EVENT_DEFAULTS["THOUGHT_BOUNDARY"],
                "status": "INTENT_ONLY_NO_SOURCE_WITHHOLDING",
                "cause": group["relation_to_previous"],
            })
        if "decision_latency" in micro:
            events.append({
                "kind": "RETRIEVAL_HESITATION",
                "group_index": group["index"],
                "requested_duration_s": _EVENT_DEFAULTS["RETRIEVAL_HESITATION"],
                "status": "INTENT_ONLY_NO_SOURCE_WITHHOLDING",
                "cause": "explicit conversational microbehavior",
            })
        if "repair_reset" in micro or group["role"] == "REVISION":
            events.append({
                "kind": "SELF_CORRECTION",
                "group_index": group["index"],
                "requested_duration_s": _EVENT_DEFAULTS["SELF_CORRECTION"],
                "status": "INTENT_ONLY_NO_SOURCE_WITHHOLDING",
                "cause": "explicit lexical/discourse revision",
            })
        if group["information_focus"]["contrast_strength"] > 0:
            events.append({
                "kind": "EMPHASIS_PREPARATION",
                "group_index": group["index"],
                "requested_duration_s": _EVENT_DEFAULTS["EMPHASIS_PREPARATION"],
                "status": "REPRESENTED_NOT_SEPARATELY_ACTUATED",
                "cause": "explicit contrastive information structure",
            })
        quiet = float(unit.get("quiet_intake_before_s", 0.0) or 0.0)
        if quiet > 0:
            events.append({
                "kind": "PHYSICAL_INTAKE_REQUEST",
                "group_index": group["index"],
                "requested_duration_s": round(quiet, 3),
                "status": "ROUTED_TO_RESPIRATION_ACTUATOR",
                "cause": "embodied phrase budget",
            })

    terminal_kind = "TURN_HOLD" if turn["exit_action"] == "HOLD_FLOOR" else "TURN_YIELD"
    events.append({
        "kind": terminal_kind,
        "group_index": groups[-1]["index"],
        "requested_duration_s": _EVENT_DEFAULTS[terminal_kind],
        "status": "INTERACTION_EVENT",
        "cause": turn["exit_action"],
    })

    return {
        "events": events,
        "source_withholding": False,
        "generic_pause_bucket": False,
        "actuation_policy": "typed timing intent must be routed to a qualified actuator; otherwise it remains represented and unapplied",
    }


def verify_microtiming(microtiming: Dict[str, Any]) -> bool:
    if microtiming.get("source_withholding") is not False:
        raise ValueError("microtiming reintroduced source-token withholding")
    if microtiming.get("generic_pause_bucket") is not False:
        raise ValueError("timing phenomena were flattened into generic pauses")
    allowed = set(_EVENT_DEFAULTS)
    for event in microtiming.get("events", []):
        if event.get("kind") not in allowed:
            raise ValueError("unsupported microtiming event")
        d = float(event.get("requested_duration_s", 0.0))
        if not 0.0 <= d <= 1.5:
            raise ValueError("microtiming duration outside safe representation envelope")
    return True
