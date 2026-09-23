"""Identity-safe native adapter for Mari Conversational Performance Compiler v1.

The adapter projects only the currently qualified measured same-speaker
finality/continuation coordinate into the existing one-session native trajectory.
All other planned performance coordinates remain represented but unapplied.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List

import numpy as np

from ..native_conversational_continuity_v1.planner import (
    PROFILE_SHA256,
    compile_native_continuity_plan,
    digest as native_digest,
    verify_native_continuity_plan,
)
from .compiler import (
    SCHEMA as PERFORMANCE_SCHEMA,
    STATE_ID as PERFORMANCE_STATE_ID,
    compile_performance_plan,
    digest,
    verify_performance_plan,
)

STATE_ID = "MARI_CONVERSATIONAL_PERFORMANCE_NATIVE_ADAPTER_v1"
SCHEMA = "mari-conversational-performance-native-plan/1.0"


def _cosine_bump(frame: int, start: int, center: int, end: int) -> float:
    if frame < start or frame >= end or end <= start:
        return 0.0
    if frame <= center:
        denom = max(1, center - start)
        t = (frame - start) / denom
        return 0.5 - 0.5 * math.cos(math.pi * t)
    denom = max(1, end - center)
    t = (frame - center) / denom
    return 0.5 + 0.5 * math.cos(math.pi * t)


def _project_discourse_finality(
    native: Dict[str, Any],
    performance: Dict[str, Any],
) -> Dict[str, Any]:
    out = copy.deepcopy(native)
    frame_count = int(out["trajectory"]["frame_count"])
    weights = np.zeros((frame_count,), dtype=np.float32)
    release = out["release"]["earliest_frames"]
    native_units = {u["index"]: u for u in out["units"]}
    groups = {g["index"]: g for g in performance["thought_groups"]}
    segments: List[Dict[str, Any]] = []

    for target in performance["group_targets"]:
        group = groups[target["group_index"]]
        unit = native_units[group["unit_index"]]
        token_end = int(unit["token_span"][1])
        if token_end >= len(release):
            raise ValueError("discourse boundary exceeds native release schedule")
        center = int(release[token_end])
        start = max(0, center - 10)
        end = min(frame_count, center + 6)
        amplitude = float(target["native_finality_target"])
        if abs(amplitude) > 0.30:
            raise ValueError("performance target outside adapter envelope")
        for f in range(start, end):
            weights[f] += amplitude * _cosine_bump(f, start, center, end)
        segments.append({
            "group_index": group["index"],
            "unit_index": group["unit_index"],
            "role": group["role"],
            "relation": group["relation_to_previous"],
            "source_token_span": list(unit["token_span"]),
            "native_frame_window": [start, end],
            "center_frame": center,
            "target": round(amplitude, 6),
            "projection": "smooth local boundary-contour actuation on measured same-speaker axis",
        })

    weights = np.clip(weights, -0.45, 0.45)
    out["trajectory"]["weights"] = weights.tolist()
    out["trajectory"]["onset"] = [0.0 for _ in out["trajectory"]["onset"]]
    out["trajectory"]["segments"] = segments
    out["trajectory"]["mode"] = "CPC_V1_DISCOURSE_BOUNDARY_FINALITY_ONLY"
    out["trajectory"]["all_injected_weights_zero"] = bool(np.count_nonzero(weights) == 0)
    out["trajectory"]["qualified_source"] = "measured same-speaker finality/continuation direction already present in native continuity layer"
    out["trajectory"]["non_actuated_performance_channels"] = [
        k for k, status in performance["performance_space"]["actuation"].items()
        if status != "QUALIFIED_MEASURED_SAME_SPEAKER_AXIS"
    ]

    for hold in out["release"]["holds"]:
        if hold.get("applied_frames") != 0:
            raise ValueError("native base unexpectedly reintroduced source holds")
    out["release"]["policy"] = "NO_EXTRA_SOURCE_WITHHOLDING_AFTER_STARVATION_EVIDENCE"

    out["laws"].update({
        "categorical_thought_prosody": False,
        "categorical_relationship_prosody": False,
        "emotion_preset_prosody": False,
        "raw_cross_speaker_performance_reference": False,
        "manual_performance_projection": False,
        "source_release_holds": False,
        "physical_quiet_intake_via_source_withholding": False,
        "performance_projection_is_discourse_derived": True,
        "only_qualified_native_axis_actuated": True,
    })
    out["performance_compiler"] = {
        "state_id": PERFORMANCE_STATE_ID,
        "schema": PERFORMANCE_SCHEMA,
        "plan_hash": performance["plan_hash"],
        "actuated_channel": "native_finality_continuation",
        "respiration_state": performance["respiration"]["state_id"],
        "respiration_actuated": False,
        "turn_entry_latency_s": performance["turn_controller"]["entry_latency_s"],
        "turn_entry_latency_is_external_to_waveform": True,
    }
    out["proof_ceiling"] = (
        "One-session Mari continuity is preserved while discourse-derived local finality/continuation "
        "targets are projected through the existing measured same-speaker axis. Tempo, energy, pitch, "
        "articulation, vocal weight, physical respiration, and interaction latency are not falsely "
        "claimed as native acoustic actuators."
    )
    bare = {k: v for k, v in out.items() if k != "plan_hash"}
    out["plan_hash"] = native_digest(bare)
    verify_native_continuity_plan(out)
    return out


def compile_native_performance(
    conversation_plan: Dict[str, Any],
    tokenizer: Any,
    *,
    performance_plan: Dict[str, Any] | None = None,
    seed: int = 160001,
) -> Dict[str, Any]:
    performance = performance_plan or compile_performance_plan(conversation_plan)
    verify_performance_plan(performance)
    if performance["source_conversation_plan_hash"] != conversation_plan["plan_hash"]:
        raise ValueError("performance plan belongs to another conversation plan")

    native = compile_native_continuity_plan(conversation_plan, tokenizer, seed=seed)
    native = _project_discourse_finality(native, performance)

    plan = {
        "schema": SCHEMA,
        "state_id": STATE_ID,
        "source_conversation_plan_hash": conversation_plan["plan_hash"],
        "performance_plan_hash": performance["plan_hash"],
        "identity": {
            "subject": "Mari404",
            "profile_sha256": PROFILE_SHA256,
            "identity_changed": False,
        },
        "performance_plan": performance,
        "native_plan": native,
        "execution_contract": {
            "decoder_sessions": 1,
            "model_invocations": 1,
            "unit_restarts": 0,
            "codec_state_resets": 0,
            "kv_cache_resets": 0,
            "waveform_stitching": False,
            "source_release_holds": False,
            "raw_cross_speaker_reference": False,
            "generic_tts_fallback": False,
            "fake_breath_audio": False,
            "respiration_actuated": False,
        },
        "proof_ceiling": native["proof_ceiling"],
    }
    plan["plan_hash"] = digest(plan)
    verify_native_performance_plan(plan)
    return plan


def verify_native_performance_plan(plan: Dict[str, Any]) -> bool:
    if plan.get("schema") != SCHEMA or plan.get("state_id") != STATE_ID:
        raise ValueError("native performance plan schema mismatch")
    supplied = plan.get("plan_hash")
    bare = {k: v for k, v in plan.items() if k != "plan_hash"}
    if supplied != digest(bare):
        raise ValueError("native performance plan hash mismatch")
    performance = plan["performance_plan"]
    verify_performance_plan(performance)
    if performance["plan_hash"] != plan["performance_plan_hash"]:
        raise ValueError("performance plan hash mismatch")
    native = plan["native_plan"]
    verify_native_continuity_plan(native)
    if native["performance_compiler"]["plan_hash"] != performance["plan_hash"]:
        raise ValueError("native projection does not bind the performance plan")
    if native["identity"]["profile_sha256"] != PROFILE_SHA256:
        raise ValueError("Mari native profile changed")
    if any(h.get("applied_frames") != 0 for h in native["release"]["holds"]):
        raise ValueError("source withholding re-entered through performance adapter")
    contract = plan["execution_contract"]
    if contract["decoder_sessions"] != 1 or contract["model_invocations"] != 1:
        raise ValueError("native continuity contract changed")
    for key in (
        "waveform_stitching", "source_release_holds", "raw_cross_speaker_reference",
        "generic_tts_fallback", "fake_breath_audio", "respiration_actuated",
    ):
        if contract.get(key) is not False:
            raise ValueError("unqualified execution path opened: " + key)
    weights = np.asarray(native["trajectory"]["weights"], dtype=np.float32)
    if not np.isfinite(weights).all() or np.max(np.abs(weights)) > 0.45:
        raise ValueError("projected trajectory outside CPC envelope")
    return True
