"""Render Mari Conversational Performance Compiler v1 plans through the verified native route."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from ..generative_v5.native import render as native_render
from ..native_conversational_continuity_v1.planner import (
    PROFILE_SHA256,
    materialize_packets,
)
from .adapter import STATE_ID, verify_native_performance_plan

RECEIPT_SCHEMA = "mari-conversational-performance-native-receipt/1.0"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_native_performance(
    root: str | Path,
    plan: Dict[str, Any],
    output: str | Path,
) -> Dict[str, Any]:
    verify_native_performance_plan(plan)
    native = plan["native_plan"]
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)

    packet_dir = out.parent / (out.stem + "_cpc_packets")
    packets = materialize_packets(native, packet_dir)
    record = native_render(
        root,
        native["text"],
        out,
        seed=native["seed"],
        trajectory=packets["trajectory"]["path"],
        incremental_text=True,
        text_release=packets["release"]["path"],
        profile_mode="xvector",
    )

    env = record.get("explicit_environment", {})
    required = {"MARI_TRAJECTORY", "MARI_TEXT_RELEASE", "MARI_TEXT_RELEASE_AUDIT", "QWEN_TTS_STREAM_LAYOUT"}
    if not required.issubset(env):
        raise RuntimeError("CPC native inputs did not all enter renderer")
    if record.get("prose_instructions") is not False:
        raise RuntimeError("prose acting instruction channel became active")
    if record.get("profile_mode") != "xvector" or record.get("profile_file_sha256") != PROFILE_SHA256:
        raise RuntimeError("selected Mari profile changed")
    if record.get("performance_reference") is not None:
        raise RuntimeError("raw cross-speaker performance reference entered CPC route")
    if not record.get("text_consumption", {}).get("no_early_consumption"):
        raise RuntimeError("source-bound text release was not verified")
    if record.get("trajectory_sha256") != packets["trajectory"]["sha256"]:
        raise RuntimeError("CPC trajectory receipt mismatch")

    perf = plan["performance_plan"]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "state_id": STATE_ID,
        "status": "CPC_V1_NATIVE_RENDER_COMPLETE",
        "plan_hash": plan["plan_hash"],
        "performance_plan_hash": perf["plan_hash"],
        "native_plan_hash": native["plan_hash"],
        "output": record["audio"],
        "identity": {
            "profile_mode": "xvector",
            "profile_file_sha256": record["profile_file_sha256"],
            "identity_changed": False,
        },
        "native_execution": {
            "decoder_sessions": 1,
            "model_invocations": 1,
            "unit_restarts": 0,
            "codec_state_resets": 0,
            "kv_cache_resets": 0,
            "waveform_stitching": False,
            "speaker_profile_constant": True,
        },
        "conversation": {
            "communicative_state": perf["communicative_state"],
            "turn_controller": perf["turn_controller"],
            "thought_groups": perf["thought_groups"],
        },
        "performance": {
            "trajectory_mode": native["trajectory"]["mode"],
            "actuated_channel": native["performance_compiler"]["actuated_channel"],
            "non_actuated_channels": native["trajectory"]["non_actuated_performance_channels"],
            "microtiming": perf["microtiming"],
            "respiration": perf["respiration"],
        },
        "causal_inputs": {
            "text_release_sha256": packets["release"]["sha256"],
            "trajectory_sha256": packets["trajectory"]["sha256"],
            "text_consumption": record["text_consumption"],
        },
        "render_policy": {
            "generic_tts_fallback": False,
            "prose_style_instruction": False,
            "random_humanization": False,
            "synthetic_breath_audio": False,
            "raw_cross_speaker_reference": False,
            "categorical_state_to_prosody": False,
            "source_release_holds": False,
        },
        "native_record": record,
        "proof_ceiling": (
            "This receipt proves execution of the CPC v1 identity-safe native route and exact causal inputs. "
            "It does not prove that the resulting performance is perceptually natural or final."
        ),
    }
    rp = out.with_suffix(".cpc.receipt.json")
    rp.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["receipt_path"] = str(rp)
    receipt["receipt_sha256"] = _sha(rp)
    return receipt
