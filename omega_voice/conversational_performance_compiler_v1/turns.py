"""Conversational turn-transition controller for Mari.

This controller represents interaction timing and floor state. It does not
implement acoustic pauses by withholding source tokens.
"""
from __future__ import annotations

from typing import Any, Dict, List


def compile_turn_controller(plan: Dict[str, Any], communicative: Dict[str, Any]) -> Dict[str, Any]:
    before = plan["state_before"]["interaction"]
    base_latency = float(plan["turn_taking"]["response_latency_s"])
    urgency = float(plan["state_observed"]["listener_model"].get("urgency", 0.0))
    interruption_risk = float(plan["state_observed"]["listener_model"].get("interruption_risk", 0.0))

    entry_latency = max(0.04, min(0.80, base_latency - 0.05 * urgency + 0.04 * interruption_risk))

    if communicative["completion"] == "OPEN":
        exit_action = "HOLD_FLOOR"
    else:
        exit_action = "YIELD_FLOOR"

    if communicative["response_expected"]:
        exit_action = "YIELD_FOR_RESPONSE"

    transitions: List[Dict[str, Any]] = [
        {"from": before.get("phase", "listening"), "to": "ENTRY_PENDING", "cause": "listener turn observed"},
        {"from": "ENTRY_PENDING", "to": "SPEAKING", "cause": "response entry latency elapsed"},
        {"from": "SPEAKING", "to": exit_action, "cause": "utterance completion state"},
    ]
    if exit_action.startswith("YIELD"):
        transitions.append({"from": exit_action, "to": "LISTENING", "cause": "floor released"})

    interrupted = before.get("phase") == "interrupted" or bool(before.get("interrupted_intention"))

    return {
        "entry_latency_s": round(entry_latency, 3),
        "source_turn_strategy": plan["turn_taking"]["floor_strategy"],
        "entry_mode": "RESUME" if interrupted else "NEW_TURN",
        "exit_action": exit_action,
        "response_expected": communicative["response_expected"],
        "interruption_risk": round(interruption_risk, 4),
        "transitions": transitions,
        "waveform_pause_required": False,
        "law": "turn-entry latency is interaction scheduling, not a synthesized silence or delayed lexical source release",
    }


def verify_turn_controller(controller: Dict[str, Any]) -> bool:
    if not 0.0 <= float(controller["entry_latency_s"]) <= 0.8:
        raise ValueError("turn entry latency out of bounds")
    if controller["exit_action"] not in {"HOLD_FLOOR", "YIELD_FLOOR", "YIELD_FOR_RESPONSE"}:
        raise ValueError("invalid turn exit action")
    if controller.get("waveform_pause_required") is not False:
        raise ValueError("turn controller attempted to encode interaction latency as waveform silence")
    return True
