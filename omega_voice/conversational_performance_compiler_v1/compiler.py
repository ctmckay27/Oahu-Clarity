"""Mari Conversational Performance Compiler v1.

Transforms an already source-grounded conversational plan into a rich,
receiver-facing performance plan. The compiler separates:
  conversation -> communicative action -> discourse/information structure
  -> continuous performance intent -> qualified acoustic projection.

It does not use categorical affect or relationship labels as direct acoustic
instructions. Only qualified actuator dimensions may leave the representation.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Tuple

from ..conversational_v6.runtime import verify_plan as verify_conversation_plan
from .discourse import compile_thought_groups, resolve_communicative_state
from .microtiming import compile_microtiming, verify_microtiming
from .respiration import compile_respiration_contract, verify_respiration_contract
from .turns import compile_turn_controller, verify_turn_controller

STATE_ID = "MARI_CONVERSATIONAL_PERFORMANCE_COMPILER_v1"
SCHEMA = "mari-conversational-performance-plan/1.0"
PROFILE_SHA256 = "9c49146ae2bd3c5a32c686ad1705fd912391af4a39db0d63e9dc353640d304fa"
ANCHOR_SHA256 = "73bb894fc75e18dddf76e3327f3bf48e161b7dd8b30fbe4949c3ac7318c29567"
SAMPLES = 96


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _smoothstep(x: float) -> float:
    x = _clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _interpolate(knots: List[Tuple[float, float]], n: int = SAMPLES) -> List[float]:
    """Piecewise smooth interpolation over normalized utterance progress."""
    if not knots:
        return [0.0] * n
    knots = sorted((float(x), float(y)) for x, y in knots)
    if knots[0][0] > 0.0:
        knots.insert(0, (0.0, knots[0][1]))
    if knots[-1][0] < 1.0:
        knots.append((1.0, knots[-1][1]))
    out: List[float] = []
    j = 0
    for i in range(n):
        x = i / max(1, n - 1)
        while j + 1 < len(knots) and x > knots[j + 1][0]:
            j += 1
        if j + 1 >= len(knots):
            out.append(knots[-1][1])
            continue
        x0, y0 = knots[j]
        x1, y1 = knots[j + 1]
        if x1 <= x0:
            out.append(y1)
            continue
        t = _smoothstep((x - x0) / (x1 - x0))
        out.append(y0 + (y1 - y0) * t)
    return [round(v, 6) for v in out]


def _group_targets(
    groups: List[Dict[str, Any]],
    communicative: Dict[str, Any],
    turn: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Derive performance intent from discourse consequences, not affect presets."""
    targets: List[Dict[str, Any]] = []
    total_words = max(1, groups[-1]["word_span"][1])

    for group in groups:
        start, end = group["word_span"]
        midpoint = ((start + end) / 2.0) / total_words
        boundary = end / total_words
        terminal = group["closure"] == "TERMINAL"
        role = group["role"]
        focus = float(group["information_focus"]["focus_strength"])

        if terminal:
            if turn["exit_action"] == "HOLD_FLOOR":
                finality = -0.12
            elif communicative["act"] == "repair":
                finality = 0.12
            elif communicative["response_expected"]:
                finality = 0.08
            else:
                finality = 0.18
        elif role in {"REVISION", "CONTRAST"}:
            finality = -0.18
        else:
            finality = -0.10

        if role == "REVISION":
            tempo = -0.18
            articulation = 0.22
        elif role == "QUESTION":
            tempo = -0.04
            articulation = 0.12
        elif role == "CONTRAST":
            tempo = -0.08
            articulation = 0.18
        else:
            tempo = 0.02
            articulation = 0.06

        energy = _clamp(0.12 * focus, -0.25, 0.25)
        pitch_motion = _clamp(0.16 * focus, -0.30, 0.30)
        vocal_weight = 0.0

        targets.append({
            "group_index": group["index"],
            "progress_midpoint": round(midpoint, 6),
            "progress_boundary": round(boundary, 6),
            "native_finality_target": round(finality, 6),
            "tempo_tendency": round(tempo, 6),
            "energy_tendency": round(energy, 6),
            "articulation_tendency": round(articulation, 6),
            "pitch_movement_tendency": round(pitch_motion, 6),
            "vocal_weight_tendency": round(vocal_weight, 6),
            "focus_strength": round(focus, 6),
            "derivation": {
                "role": role,
                "closure": group["closure"],
                "relation": group["relation_to_previous"],
                "turn_exit": turn["exit_action"],
                "communicative_act": communicative["act"],
            },
        })
    return targets


def _continuous_space(targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    channels = {
        "native_finality_continuation": [],
        "tempo_tendency": [],
        "energy_tendency": [],
        "articulation_tendency": [],
        "pitch_movement_tendency": [],
        "vocal_weight_tendency": [],
        "focus_strength": [],
    }
    field_for_channel = {
        "native_finality_continuation": "native_finality_target",
        "tempo_tendency": "tempo_tendency",
        "energy_tendency": "energy_tendency",
        "articulation_tendency": "articulation_tendency",
        "pitch_movement_tendency": "pitch_movement_tendency",
        "vocal_weight_tendency": "vocal_weight_tendency",
        "focus_strength": "focus_strength",
    }
    for channel, field in field_for_channel.items():
        knots = [(0.0, targets[0][field])]
        for t in targets:
            knots.append((t["progress_midpoint"], t[field]))
            knots.append((t["progress_boundary"], t[field]))
        channels[channel] = _interpolate(knots)

    return {
        "domain": "normalized_utterance_progress",
        "sample_count": SAMPLES,
        "channels": channels,
        "actuation": {
            "native_finality_continuation": "QUALIFIED_MEASURED_SAME_SPEAKER_AXIS",
            "tempo_tendency": "REPRESENTED_NOT_ACTUATED",
            "energy_tendency": "REPRESENTED_NOT_ACTUATED",
            "articulation_tendency": "REPRESENTED_NOT_ACTUATED",
            "pitch_movement_tendency": "REPRESENTED_NOT_ACTUATED",
            "vocal_weight_tendency": "REPRESENTED_NOT_ACTUATED",
            "focus_strength": "REPRESENTED_INFORMATION_STRUCTURE_NOT_DIRECT_ACTUATOR",
        },
    }


def compile_performance_plan(conversation_plan: Dict[str, Any]) -> Dict[str, Any]:
    verify_conversation_plan(conversation_plan)
    identity = conversation_plan["state_before"]["identity"]
    if identity.get("profile_sha256") != PROFILE_SHA256 or identity.get("anchor_sha256") != ANCHOR_SHA256:
        raise ValueError("conversation plan does not target selected Mari identity")

    communicative = resolve_communicative_state(conversation_plan)
    groups = compile_thought_groups(conversation_plan, communicative)
    if not groups:
        raise ValueError("performance compiler requires at least one discourse group")

    turn = compile_turn_controller(conversation_plan, communicative)
    microtiming = compile_microtiming(conversation_plan, groups, turn)
    respiration = compile_respiration_contract(microtiming)
    group_targets = _group_targets(groups, communicative, turn)
    performance_space = _continuous_space(group_targets)

    plan = {
        "schema": SCHEMA,
        "state_id": STATE_ID,
        "source_conversation_plan_hash": conversation_plan["plan_hash"],
        "session_id": conversation_plan["session_id"],
        "turn": conversation_plan["turn"],
        "listener_id": conversation_plan["listener_id"],
        "text": conversation_plan["response_text"],
        "identity": {
            "subject": "Mari404",
            "anchor_sha256": ANCHOR_SHA256,
            "profile_sha256": PROFILE_SHA256,
            "identity_changed": False,
            "law": "Mari becomes otherwise without becoming someone else",
        },
        "communicative_state": communicative,
        "thought_groups": groups,
        "group_targets": group_targets,
        "performance_space": performance_space,
        "microtiming": microtiming,
        "respiration": respiration,
        "turn_controller": turn,
        "actuator_firewall": {
            "categorical_thought_to_acoustics": False,
            "relationship_to_acoustics": False,
            "emotion_preset_to_acoustics": False,
            "raw_cross_speaker_reference": False,
            "generic_tts_fallback": False,
            "random_humanization": False,
            "source_token_withholding_for_timing": False,
            "fake_breath_audio": False,
            "waveform_stitching": False,
            "qualified_native_dimensions": ["measured_same_speaker_finality_continuation_axis"],
        },
        "learning_interface": {
            "ordinary_reaction_evidence_supported": True,
            "numeric_rating_required": False,
            "automatic_self_modification": False,
            "update_rule": "reaction -> localize candidate mismatch -> preserve evidence -> change only after supported mapping is selected",
        },
        "proof_ceiling": (
            "This plan compiles conversational meaning into typed discourse, timing, turn, respiration, "
            "and continuous performance coordinates. Only the measured same-speaker finality/continuation "
            "axis is currently qualified for native acoustic actuation; all other coordinates remain explicit "
            "but unrealized until independently qualified."
        ),
    }
    plan["plan_hash"] = digest(plan)
    verify_performance_plan(plan)
    return plan


def verify_performance_plan(plan: Dict[str, Any]) -> bool:
    if plan.get("schema") != SCHEMA or plan.get("state_id") != STATE_ID:
        raise ValueError("performance plan schema mismatch")
    supplied = plan.get("plan_hash")
    bare = {k: v for k, v in plan.items() if k != "plan_hash"}
    if supplied != digest(bare):
        raise ValueError("performance plan hash mismatch")
    identity = plan["identity"]
    if identity["profile_sha256"] != PROFILE_SHA256 or identity["anchor_sha256"] != ANCHOR_SHA256:
        raise ValueError("Mari identity mismatch")
    if identity.get("identity_changed") is not False:
        raise ValueError("Mari identity changed")
    verify_turn_controller(plan["turn_controller"])
    verify_microtiming(plan["microtiming"])
    verify_respiration_contract(plan["respiration"])
    firewall = plan["actuator_firewall"]
    forbidden = [
        "categorical_thought_to_acoustics", "relationship_to_acoustics",
        "emotion_preset_to_acoustics", "raw_cross_speaker_reference",
        "generic_tts_fallback", "random_humanization",
        "source_token_withholding_for_timing", "fake_breath_audio",
        "waveform_stitching",
    ]
    if any(firewall.get(k) is not False for k in forbidden):
        raise ValueError("actuator firewall opened an unqualified path")
    channels = plan["performance_space"]["channels"]
    n = plan["performance_space"]["sample_count"]
    if not channels or any(len(v) != n for v in channels.values()):
        raise ValueError("continuous performance channel shape mismatch")
    finality = channels["native_finality_continuation"]
    if any(not math.isfinite(float(v)) or abs(float(v)) > 0.30 for v in finality):
        raise ValueError("native finality intent outside bounded planning envelope")
    return True
