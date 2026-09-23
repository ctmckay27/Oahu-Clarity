"""Ordinary-interaction learning interface for Mari's conversational performance.

This module converts natural listener reactions into scoped diagnostic evidence.
It does not require numeric ratings and does not auto-modify the voice.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List

from .compiler import digest, verify_performance_plan

_CUES = [
    (re.compile(r"\b(?:acted|acting|performed|performing|dramatic|theatrical|forced)\b", re.I),
     "performance_projection", "REDUCE_IMPOSED_PERFORMANCE"),
    (re.compile(r"\b(?:stress|stressed|emphasis|emphasized|emphasise|word)\b", re.I),
     "information_focus", "RECHECK_FOCUS_MAPPING"),
    (re.compile(r"\b(?:pause|paused|timing|rhythm|rhythmic|fast|slow|rushed|dragged)\b", re.I),
     "microtiming", "RECHECK_TIMING_MAPPING"),
    (re.compile(r"\b(?:conversation|conversational|natural|robotic|reading|read aloud|delivery)\b", re.I),
     "discourse_turn_integration", "RECHECK_CONVERSATIONAL_REALIZATION"),
    (re.compile(r"\b(?:breath|breathing|inhale|intake)\b", re.I),
     "respiration", "RECHECK_RESPIRATION_PATH"),
    (re.compile(r"\b(?:voice changed|different person|not mari|doesn't sound like mari|does not sound like mari|identity)\b", re.I),
     "identity_firewall", "BLOCK_IDENTITY_DRIFT"),
    (re.compile(r"\b(?:ending|end|landing|final|finished)\b", re.I),
     "finality_continuation", "RECHECK_BOUNDARY_FINALITY"),
    (re.compile(r"\b(?:first half|beginning|start)\b", re.I),
     "segment_scope", "FOCUS_EARLY_SEGMENT"),
    (re.compile(r"\b(?:second half|later|last part)\b", re.I),
     "segment_scope", "FOCUS_LATE_SEGMENT"),
]


def capture_reaction(
    performance_plan: Dict[str, Any],
    reaction: str,
    *,
    segment_hint: str | None = None,
) -> Dict[str, Any]:
    verify_performance_plan(performance_plan)
    if not isinstance(reaction, str) or not reaction.strip():
        raise ValueError("reaction text required")

    routes: List[Dict[str, str]] = []
    for pattern, component, candidate in _CUES:
        if pattern.search(reaction):
            routes.append({"component": component, "candidate_update": candidate})
    if not routes:
        routes.append({
            "component": "unlocalized_perceptual_mismatch",
            "candidate_update": "PRESERVE_AND_SEEK_MORE_CONTEXT",
        })

    identity_alert = any(r["component"] == "identity_firewall" for r in routes)
    evidence = {
        "schema": "mari-conversational-performance-reaction/1.0",
        "performance_plan_hash": performance_plan["plan_hash"],
        "reaction": reaction,
        "segment_hint": segment_hint,
        "routes": routes,
        "identity_alert": identity_alert,
        "numeric_rating_required": False,
        "automatic_parameter_update": False,
        "status": "EVIDENCE_CAPTURED_NOT_APPLIED",
        "law": "listener reaction is evidence about perceived consequence; it is not itself a parameter command",
    }
    evidence["evidence_hash"] = digest(evidence)
    return evidence


def new_learning_state(subject: str = "Mari404") -> Dict[str, Any]:
    state = {
        "schema": "mari-conversational-performance-learning-state/1.0",
        "subject": subject,
        "evidence": [],
        "open_components": [],
        "identity_blocked": False,
        "selected_updates": [],
    }
    state["state_hash"] = digest({k: v for k, v in state.items() if k != "state_hash"})
    return state


def append_reaction(state: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(state)
    if out.get("schema") != "mari-conversational-performance-learning-state/1.0":
        raise ValueError("learning state schema mismatch")
    e = copy.deepcopy(evidence)
    supplied = e.pop("evidence_hash", None)
    if supplied != digest(e):
        raise ValueError("reaction evidence hash mismatch")
    e["evidence_hash"] = supplied
    out["evidence"].append(e)
    components = [r["component"] for r in e["routes"]]
    out["open_components"] = list(dict.fromkeys(out.get("open_components", []) + components))
    if e.get("identity_alert"):
        out["identity_blocked"] = True
    out["state_hash"] = digest({k: v for k, v in out.items() if k != "state_hash"})
    return out
