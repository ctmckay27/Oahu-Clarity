"""Respiration/recovery actuator contract for Mari.

The current native renderer has no qualified non-starving physical intake
mechanism. This module still builds the actuator boundary now: it receives typed
respiration intent, rejects unsafe source-withholding implementations, records
what remains unrealized, and provides a qualification interface for a future
native actuator without conflating "requested" with "executed".
"""
from __future__ import annotations

from typing import Any, Dict, List

STATE = "MARI_NATIVE_RESPIRATION_ACTUATOR_v1"
QUALIFIED_MECHANISMS = set()
FORBIDDEN_MECHANISMS = {
    "source_token_withholding",
    "waveform_stitching",
    "fake_breath_clip",
    "decoder_restart",
    "speaker_profile_substitution",
}


def compile_respiration_contract(microtiming: Dict[str, Any]) -> Dict[str, Any]:
    requests: List[Dict[str, Any]] = []
    for event in microtiming.get("events", []):
        if event.get("kind") != "PHYSICAL_INTAKE_REQUEST":
            continue
        requests.append({
            "group_index": event["group_index"],
            "requested_duration_s": event["requested_duration_s"],
            "mechanism": None,
            "actuated": False,
            "status": "REQUESTED__NO_QUALIFIED_NATIVE_ACTUATOR",
            "reason": "source withholding is starvation-unsafe and no independent native pause/intake actuator has been qualified",
        })

    return {
        "state_id": STATE,
        "requests": requests,
        "qualified_mechanisms": sorted(QUALIFIED_MECHANISMS),
        "forbidden_mechanisms": sorted(FORBIDDEN_MECHANISMS),
        "source_tokens_withheld": False,
        "waveform_stitched": False,
        "fake_breath_audio": False,
        "proof_ceiling": "Respiration intent is preserved and routed, but physical intake timing is not claimed executed until a non-starving native actuator passes qualification.",
    }


def qualification_candidate(
    *,
    mechanism: str,
    identity_preserved: bool,
    one_decoder_session: bool,
    source_consumption_preserved: bool,
    audible_timing_effect: bool,
    intelligibility_preserved: bool,
    evidence_id: str,
) -> Dict[str, Any]:
    if not mechanism or mechanism in FORBIDDEN_MECHANISMS:
        status = "REJECTED"
    elif all((identity_preserved, one_decoder_session, source_consumption_preserved,
              audible_timing_effect, intelligibility_preserved, bool(evidence_id))):
        status = "CANDIDATE_PASSES_STRUCTURAL_GATE__NOT_AUTO_PROMOTED"
    else:
        status = "UNQUALIFIED"
    return {
        "mechanism": mechanism,
        "status": status,
        "identity_preserved": bool(identity_preserved),
        "one_decoder_session": bool(one_decoder_session),
        "source_consumption_preserved": bool(source_consumption_preserved),
        "audible_timing_effect": bool(audible_timing_effect),
        "intelligibility_preserved": bool(intelligibility_preserved),
        "evidence_id": evidence_id,
    }


def verify_respiration_contract(contract: Dict[str, Any]) -> bool:
    if contract.get("state_id") != STATE:
        raise ValueError("respiration actuator state mismatch")
    if contract.get("source_tokens_withheld") is not False:
        raise ValueError("unsafe respiration source withholding")
    if contract.get("waveform_stitched") is not False:
        raise ValueError("respiration via waveform stitching is forbidden")
    if contract.get("fake_breath_audio") is not False:
        raise ValueError("fake breath audio is forbidden")
    for req in contract.get("requests", []):
        if req.get("actuated") is not False or req.get("mechanism") is not None:
            raise ValueError("unqualified respiration request was marked actuated")
    return True
