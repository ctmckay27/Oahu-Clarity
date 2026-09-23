"""Mari performance-transfer renderer."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

from .reference import validate_performance_reference
from ..generative_v5.native import render as native_render
from ..causal_v4.runtime import PROFILE

RECEIPT_SCHEMA="mari-performance-transfer-receipt/1.0"

def render_transfer(
    root: str|Path,
    *,
    text: str,
    reference: Dict[str,Any],
    output: str|Path,
    seed: int,
) -> Dict[str,Any]:
    reference=validate_performance_reference(reference)
    record=native_render(
        root,text,output,seed=seed,
        performance_reference=reference,
        profile_mode="xvector",
    )
    if record.get("profile_file_sha256")!=PROFILE:
        raise RuntimeError("Mari profile changed during performance transfer")
    if record.get("speaker_identity_authority")!="MARI_VOICE_V1_PROFILE_ONLY":
        raise RuntimeError("speaker identity authority escaped Mari profile")
    observed=record.get("performance_reference")
    if not observed or observed.get("sha256")!=reference["sha256"]:
        raise RuntimeError("performance reference receipt mismatch")
    receipt={
        "schema":RECEIPT_SCHEMA,
        "status":"PERFORMANCE_TRANSFER_RENDER_COMPLETE",
        "text":text,
        "seed":seed,
        "identity":{
            "speaker_profile_sha256":record["profile_file_sha256"],
            "authority":"MARI_VOICE_V1_PROFILE_ONLY",
            "donor_identity_authority":False,
        },
        "performance_reference":reference,
        "output":record["audio"],
        "render_policy":{
            "generic_tts_fallback":False,
            "prose_acting_instruction":False,
            "random_humanization":False,
            "trajectory":False,
            "incremental_text":False,
            "speaker_profile_substitution":False,
        },
        "native_record":record,
        "proof_ceiling":"This proves execution of cross-speaker performance conditioning while retaining the selected Mari x-vector input. Human listening plus speaker-similarity evaluation determine whether performance improved without identity drift.",
    }
    rp=Path(output).with_suffix(".performance-transfer.receipt.json")
    rp.write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt
