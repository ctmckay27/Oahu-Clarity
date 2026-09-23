"""End-to-end turn runtime for Mari Conversational Performance Compiler v1.

Rendering is not treated as delivery. The conversational state advances only
after the caller reports what was actually delivered to the listener.
"""
from __future__ import annotations

import copy
from typing import Any, Dict

from ..conversational_v6 import commit_delivery, plan_turn, validate_state
from .adapter import compile_native_performance, verify_native_performance_plan
from .compiler import compile_performance_plan
from .learning import append_reaction, capture_reaction, new_learning_state
from .renderer import render_native_performance


def plan_native_turn(
    state: Dict[str, Any],
    listener_text: str,
    response_text: str,
    tokenizer: Any,
    *,
    observation: Dict[str, Any] | None = None,
    intent: Dict[str, Any] | None = None,
    seed: int = 160001,
) -> Dict[str, Any]:
    validate_state(state)
    conversation = plan_turn(
        state,
        listener_text,
        response_text,
        observation=observation,
        intent=intent,
    )
    performance = compile_performance_plan(conversation)
    native = compile_native_performance(
        conversation,
        tokenizer,
        performance_plan=performance,
        seed=seed,
    )
    bundle = {
        "schema": "mari-cpc-turn-bundle/1.0",
        "conversation": conversation,
        "performance": performance,
        "native": native,
        "delivery_status": "PLANNED_NOT_RENDERED_NOT_DELIVERED",
        "interaction_schedule": {
            "entry_latency_s": performance["turn_controller"]["entry_latency_s"],
            "entry_mode": performance["turn_controller"]["entry_mode"],
            "exit_action": performance["turn_controller"]["exit_action"],
            "entry_latency_is_receiver_timing_not_waveform_silence": True,
        },
    }
    return bundle


def render_planned_turn(
    root: str,
    bundle: Dict[str, Any],
    output: str,
) -> Dict[str, Any]:
    if bundle.get("schema") != "mari-cpc-turn-bundle/1.0":
        raise ValueError("turn bundle schema mismatch")
    verify_native_performance_plan(bundle["native"])
    receipt = render_native_performance(root, bundle["native"], output)
    out = copy.deepcopy(bundle)
    out["render_receipt"] = receipt
    out["delivery_status"] = "RENDERED_NOT_DELIVERED"
    return out


def commit_actual_delivery(
    prior_state: Dict[str, Any],
    bundle: Dict[str, Any],
    *,
    delivered_word_count: int | None = None,
    interrupted: bool = False,
) -> Dict[str, Any]:
    """Advance state only after external delivery is established by the caller."""
    if bundle.get("schema") != "mari-cpc-turn-bundle/1.0":
        raise ValueError("turn bundle schema mismatch")
    if bundle.get("delivery_status") not in {"RENDERED_NOT_DELIVERED", "DELIVERY_EXTERNALLY_CONFIRMED"}:
        raise ValueError("speech may not be committed before a render exists")
    state_after = commit_delivery(
        prior_state,
        bundle["conversation"],
        delivered_word_count=delivered_word_count,
        interrupted=interrupted,
    )
    out = copy.deepcopy(bundle)
    out["state_after_delivery"] = state_after
    out["delivery_status"] = "DELIVERED_AND_COMMITTED"
    out["delivery_commit"] = {
        "interrupted": bool(interrupted),
        "delivered_word_count": delivered_word_count,
        "planned_equals_delivered_assumed": delivered_word_count is None and not interrupted,
        "receiver_effect_beyond_delivery_proven": False,
    }
    return out


def record_ordinary_reaction(
    bundle: Dict[str, Any],
    reaction: str,
    learning_state: Dict[str, Any] | None = None,
    *,
    segment_hint: str | None = None,
) -> Dict[str, Any]:
    if bundle.get("schema") != "mari-cpc-turn-bundle/1.0":
        raise ValueError("turn bundle schema mismatch")
    evidence = capture_reaction(bundle["performance"], reaction, segment_hint=segment_hint)
    state = learning_state or new_learning_state()
    updated = append_reaction(state, evidence)
    return {
        "evidence": evidence,
        "learning_state": updated,
        "applied_to_voice": False,
        "reason": "ordinary reaction is retained and localized before any bounded mapping change is selected",
    }
