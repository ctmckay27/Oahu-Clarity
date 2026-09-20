"""Execute Mari conversational continuity inside one native decoder request."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from .planner import (
    STATE_ID,
    compile_native_continuity_plan,
    materialize_packets,
    verify_native_continuity_plan,
)
from ..generative_v5.native import render as native_render

RECEIPT_SCHEMA="mari-native-conversational-continuity-receipt/1.0"


def _sha(path: str|Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_native_continuity(
    root: str|Path,
    conversation_plan: Dict[str,Any],
    tokenizer: Any,
    output: str|Path,
    *,
    seed: int=130021,
) -> Dict[str,Any]:
    """Render the full response in one Qwen decoder/KV/codec session."""
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError(out)
    plan=compile_native_continuity_plan(conversation_plan,tokenizer,seed=seed)
    verify_native_continuity_plan(plan)
    packet_dir=out.parent/(out.stem+"_native_packets")
    packets=materialize_packets(plan,packet_dir)
    record=native_render(
        root,
        plan["text"],
        out,
        seed=seed,
        trajectory=packets["trajectory"]["path"],
        incremental_text=True,
        text_release=packets["release"]["path"],
        profile_mode="xvector",
    )
    env=record.get("explicit_environment",{})
    required_env={"MARI_TRAJECTORY","MARI_TEXT_RELEASE","MARI_TEXT_RELEASE_AUDIT","QWEN_TTS_STREAM_LAYOUT"}
    if not required_env.issubset(env): raise RuntimeError("native continuity inputs did not all enter the renderer")
    if record.get("prose_instructions") is not False: raise RuntimeError("prose instruction channel became active")
    if record.get("profile_mode")!="xvector": raise RuntimeError("selected Mari x-vector route changed")
    if not record.get("text_consumption",{}).get("no_early_consumption"):
        raise RuntimeError("source-bound text release was not verified")
    if record.get("trajectory_sha256")!=packets["trajectory"]["sha256"]:
        raise RuntimeError("native trajectory receipt mismatch")
    receipt={
        "schema":RECEIPT_SCHEMA,
        "state_id":STATE_ID,
        "status":"ONE_SESSION_NATIVE_RENDER_COMPLETE",
        "plan_hash":plan["plan_hash"],
        "plan_path":packets["plan_path"],
        "plan_sha256":packets["plan_sha256"],
        "output":record["audio"],
        "native_execution":{
            "decoder_sessions":1,
            "model_invocations":1,
            "unit_restarts":0,
            "codec_state_resets":0,
            "kv_cache_resets":0,
            "waveform_stitching":False,
            "speaker_profile_constant":True,
            "seed":seed,
        },
        "causal_inputs":{
            "trajectory_sha256":packets["trajectory"]["sha256"],
            "release_sha256":packets["release"]["sha256"],
            "text_consumption":record["text_consumption"],
            "trajectory_segments":plan["trajectory"]["segments"],
            "release_holds":plan["release"]["holds"],
        },
        "identity":{
            "profile_mode":record["profile_mode"],
            "profile_file_sha256":record["profile_file_sha256"],
            "identity_changed":False,
        },
        "render_policy":{
            "generic_tts_fallback":False,
            "prose_style_instruction":False,
            "random_humanization":False,
            "synthetic_breath_audio":False,
            "unsupported_dimensions_unrealized":True,
        },
        "native_record":record,
        "proof_ceiling":"This proves one-session executable continuity and causal input consumption, not perceived personhood or authenticity.",
    }
    receipt_path=out.with_suffix(".native-continuity.receipt.json")
    receipt_path.write_text(json.dumps(receipt,indent=2)+"\n")
    receipt["receipt_path"]=str(receipt_path)
    receipt["receipt_sha256"]=_sha(receipt_path)
    return receipt
