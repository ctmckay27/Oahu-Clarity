"""Render MARI_UNPERFORMED_CONVERSATION_v1 through the verified native route."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from .planner import STATE_ID, verify_unperformed_plan
from ..native_conversational_continuity_v1.planner import (
    PROFILE_SHA256,
    materialize_packets,
)
from ..generative_v5.native import render as native_render

RECEIPT_SCHEMA="mari-unperformed-conversation-receipt/1.0"

def _sha(path: str|Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def render_unperformed(
    root: str|Path,
    plan: Dict[str,Any],
    output: str|Path,
) -> Dict[str,Any]:
    verify_unperformed_plan(plan)
    native=plan["native_plan"]
    out=Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    packet_dir=out.parent/(out.stem+"_unperformed_packets")
    packets=materialize_packets(native,packet_dir)
    record=native_render(
        root,
        native["text"],
        out,
        seed=native["seed"],
        trajectory=packets["trajectory"]["path"],
        incremental_text=True,
        text_release=packets["release"]["path"],
        profile_mode="xvector",
    )
    env=record.get("explicit_environment",{})
    required={"MARI_TRAJECTORY","MARI_TEXT_RELEASE","MARI_TEXT_RELEASE_AUDIT","QWEN_TTS_STREAM_LAYOUT"}
    if not required.issubset(env):
        raise RuntimeError("unperformed native inputs did not all enter renderer")
    if record.get("prose_instructions") is not False:
        raise RuntimeError("prose acting instruction channel became active")
    if record.get("profile_mode")!="xvector" or record.get("profile_file_sha256")!=PROFILE_SHA256:
        raise RuntimeError("selected Mari profile changed")
    if not record.get("text_consumption",{}).get("no_early_consumption"):
        raise RuntimeError("source-bound text release was not verified")
    if record.get("trajectory_sha256")!=packets["trajectory"]["sha256"]:
        raise RuntimeError("zero-actuation trajectory receipt mismatch")
    receipt={
        "schema":RECEIPT_SCHEMA,
        "state_id":STATE_ID,
        "status":"UNPERFORMED_NATIVE_RENDER_COMPLETE",
        "plan_hash":plan["plan_hash"],
        "native_plan_hash":native["plan_hash"],
        "output":record["audio"],
        "identity":{
            "profile_mode":"xvector",
            "profile_file_sha256":record["profile_file_sha256"],
            "identity_changed":False,
        },
        "native_execution":{
            "decoder_sessions":1,
            "model_invocations":1,
            "unit_restarts":0,
            "codec_state_resets":0,
            "kv_cache_resets":0,
            "waveform_stitching":False,
            "speaker_profile_constant":True,
        },
        "audibility_gate":plan["audibility_gate"],
        "disabled_performance_controls":plan["disabled_performance_controls"],
        "causal_inputs":{
            "text_release_sha256":packets["release"]["sha256"],
            "trajectory_sha256":packets["trajectory"]["sha256"],
            "trajectory_mode":native["trajectory"]["mode"],
            "all_injected_weights_zero":native["trajectory"]["all_injected_weights_zero"],
            "text_consumption":record["text_consumption"],
        },
        "render_policy":{
            "generic_tts_fallback":False,
            "prose_style_instruction":False,
            "random_humanization":False,
            "synthetic_breath_audio":False,
            "categorical_state_to_prosody":False,
            "native_uninstructed_delivery_primary":True,
        },
        "native_record":record,
        "proof_ceiling":"This proves execution of the subtractive route with frozen Mari identity and zero categorical performance actuation. It does not prove conversationality.",
    }
    rp=out.with_suffix(".unperformed.receipt.json")
    rp.write_text(json.dumps(receipt,indent=2)+"\n")
    receipt["receipt_path"]=str(rp)
    receipt["receipt_sha256"]=_sha(rp)
    return receipt
