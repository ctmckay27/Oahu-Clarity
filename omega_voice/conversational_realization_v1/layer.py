"""Mari Conversational Realization Layer v1.

Converts an upstream MARI conversational plan into a deterministic sequence of
acoustic render units. It preserves the selected Mari acoustic identity while
making thought state, turn timing, relationship state, repair, and embodied
continuity causally visible to the renderer.

This layer does not claim perceptual authenticity by construction. It produces
an inspectable coupling plan and fails closed on unsupported structure.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Dict, List, Optional

STATE_ID = "MARI_CONVERSATIONAL_REALIZATION_LAYER_v1"
SCHEMA = "mari-conversational-realization-plan/1.0"
RECEIPT_SCHEMA = "mari-conversational-realization-receipt/1.0"

ANCHOR_SHA256 = "73bb894fc75e18dddf76e3327f3bf48e161b7dd8b30fbe4949c3ac7318c29567"
PROFILE_SHA256 = "9c49146ae2bd3c5a32c686ad1705fd912391af4a39db0d63e9dc353640d304fa"

THOUGHT_CONTROLS = {
    "known":          {"rate":1.00, "volume":1.00, "pre_pause_s":0.04, "join":"flow"},
    "observing":      {"rate":0.99, "volume":0.99, "pre_pause_s":0.05, "join":"flow"},
    "judging":        {"rate":0.99, "volume":1.00, "pre_pause_s":0.05, "join":"flow"},
    "searching":      {"rate":0.94, "volume":0.97, "pre_pause_s":0.13, "join":"consider"},
    "remembering":    {"rate":0.96, "volume":0.98, "pre_pause_s":0.10, "join":"consider"},
    "deciding":       {"rate":0.96, "volume":0.99, "pre_pause_s":0.11, "join":"consider"},
    "reconsidering":  {"rate":0.95, "volume":0.98, "pre_pause_s":0.12, "join":"consider"},
    "correcting":     {"rate":1.03, "volume":1.00, "pre_pause_s":0.09, "join":"repair"},
    "realizing":      {"rate":1.02, "volume":1.01, "pre_pause_s":0.06, "join":"commit"},
    "discovering":    {"rate":0.99, "volume":1.00, "pre_pause_s":0.07, "join":"commit"},
    "suppressing":    {"rate":0.97, "volume":0.96, "pre_pause_s":0.08, "join":"hold"},
    "withholding":    {"rate":0.97, "volume":0.95, "pre_pause_s":0.09, "join":"hold"},
    "changing_tactic":{"rate":1.00, "volume":1.00, "pre_pause_s":0.08, "join":"reset"},
}


def digest(value: Any) -> str:
    payload=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo,min(hi,float(x)))


def _finite(name: str, x: Any) -> float:
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return float(x)


def _validate_conversation_plan(plan: Dict[str, Any]) -> None:
    if plan.get("schema")!="mari-conversational-plan/1.0":
        raise ValueError("unsupported conversational plan schema")
    if not isinstance(plan.get("units"),list) or not plan["units"]:
        raise ValueError("conversational plan has no units")
    contract=plan.get("renderer_contract",{})
    if contract.get("acoustic_identity_change") is not False:
        raise ValueError("acoustic identity change not allowed")
    if contract.get("generic_tts_fallback") is not False:
        raise ValueError("generic TTS fallback must be disabled")
    identity=plan.get("state_before",{}).get("identity",{})
    if identity.get("anchor_sha256")!=ANCHOR_SHA256 or identity.get("profile_sha256")!=PROFILE_SHA256:
        raise ValueError("conversation plan does not target selected Mari Voice v1 identity")


def _relationship_projection(plan: Dict[str, Any]) -> Dict[str,float]:
    rel=plan.get("state_observed",{}).get("relationship",{})
    trust=_clamp(_finite("trust",rel.get("trust",.5)),0,1)
    familiarity=_clamp(_finite("familiarity",rel.get("familiarity",.5)),0,1)
    distance=_clamp(_finite("distance",rel.get("distance",.5)),0,1)
    irritation=_clamp(_finite("irritation",rel.get("irritation",0)),0,1)
    intimacy=(trust+familiarity+(1-distance))/3
    # Relationship changes projection modestly, never identity or timbre.
    volume=_clamp(1.0 - .035*intimacy + .025*irritation,.94,1.04)
    return {"intimacy":round(intimacy,4),"volume_multiplier":round(volume,4)}


def _unit_controls(plan: Dict[str,Any], unit: Dict[str,Any], rel: Dict[str,float]) -> Dict[str,Any]:
    thought=unit.get("thought")
    if thought not in THOUGHT_CONTROLS:
        raise ValueError("unsupported thought state: "+str(thought))
    base=copy.deepcopy(THOUGHT_CONTROLS[thought])
    speech_act=unit.get("speech_act","assert")
    certainty=_clamp(_finite("certainty",unit.get("certainty",.7)),0,1)
    quiet=_clamp(_finite("quiet_intake_before_s",unit.get("quiet_intake_before_s",0.0)),0,1.5)

    if speech_act=="ask":
        base["rate"]*=.99
        base["pre_pause_s"]+=.01
    elif speech_act=="qualified_assertion":
        base["rate"]*=.985
    elif speech_act=="repair":
        base["join"]="repair"
        base["pre_pause_s"]=max(base["pre_pause_s"],.08)

    # Certainty influences commitment only through bounded timing/projection.
    if thought in {"known","realizing","discovering","judging"}:
        base["rate"]*=1.0 + .015*(certainty-.5)
        base["volume"]*=1.0 + .012*(certainty-.5)
    elif thought in {"searching","reconsidering","deciding"}:
        base["rate"]*=1.0 - .018*(1-certainty)

    base["volume"]*=rel["volume_multiplier"]
    base["pre_pause_s"]=max(base["pre_pause_s"],quiet)
    base["rate"]=round(_clamp(base["rate"],.90,1.07),4)
    base["volume"]=round(_clamp(base["volume"],.92,1.06),4)
    base["pre_pause_s"]=round(_clamp(base["pre_pause_s"],0,.7),3)

    micro=[x.get("kind") for x in unit.get("microbehavior",[]) if isinstance(x,dict)]
    base["self_monitor_event"]="repair_reset" if "repair_reset" in micro else None
    base["causal_microbehavior"]=micro
    return base


def _continuity_token(previous: Optional[Dict[str,Any]], unit: Dict[str,Any], controls: Dict[str,Any]) -> str:
    basis={
        "previous": None if previous is None else previous["continuity_token"],
        "unit_index":unit["index"],
        "text":unit["text"],
        "thought":unit["thought"],
        "controls":controls,
    }
    return digest(basis)


def build_realization_plan(
    conversation_plan: Dict[str,Any],
    *,
    base_seed: int=120031,
    prior_receipt: Optional[Dict[str,Any]]=None,
) -> Dict[str,Any]:
    _validate_conversation_plan(conversation_plan)
    if not isinstance(base_seed,int) or base_seed<0:
        raise ValueError("base_seed must be a nonnegative integer")

    relationship=_relationship_projection(conversation_plan)
    prior_token=None
    prior_reserve=None
    if prior_receipt is not None:
        if prior_receipt.get("schema")!=RECEIPT_SCHEMA:
            raise ValueError("unsupported prior realization receipt")
        if prior_receipt.get("identity",{}).get("anchor_sha256")!=ANCHOR_SHA256:
            raise ValueError("prior receipt belongs to another acoustic identity")
        prior_token=prior_receipt.get("continuity",{}).get("terminal_token")
        prior_reserve=prior_receipt.get("continuity",{}).get("terminal_breath_reserve")

    units: List[Dict[str,Any]]=[]
    previous=None
    for pos,unit in enumerate(conversation_plan["units"]):
        controls=_unit_controls(conversation_plan,unit,relationship)
        token=_continuity_token(previous,unit,controls)
        seed=(base_seed + pos*7919 + int(token[:8],16)) % 2147483647
        item={
            "index":unit["index"],
            "text":unit["text"],
            "start_word":unit["start_word"],
            "end_word":unit["end_word"],
            "thought":unit["thought"],
            "speech_act":unit.get("speech_act","assert"),
            "certainty":unit.get("certainty"),
            "reserve_after":unit.get("reserve_after"),
            "controls":controls,
            "seed":seed,
            "continuity_token":token,
            "inherits_previous_unit":pos>0,
        }
        units.append(item);previous=item

    plan={
        "schema":SCHEMA,
        "state_id":STATE_ID,
        "conversation_plan_hash":conversation_plan["plan_hash"],
        "session_id":conversation_plan["session_id"],
        "turn":conversation_plan["turn"],
        "listener_id":conversation_plan["listener_id"],
        "identity":{
            "subject":"Mari404",
            "anchor_sha256":ANCHOR_SHA256,
            "profile_sha256":PROFILE_SHA256,
            "identity_changed":False,
        },
        "turn_taking":copy.deepcopy(conversation_plan["turn_taking"]),
        "relationship_projection":relationship,
        "units":units,
        "continuity":{
            "prior_terminal_token":prior_token,
            "prior_terminal_breath_reserve":prior_reserve,
            "initial_response_latency_s":conversation_plan["turn_taking"]["response_latency_s"],
            "terminal_token":units[-1]["continuity_token"],
            "terminal_breath_reserve":conversation_plan["embodied_projection"]["breath_reserve_after_full_delivery"],
        },
        "laws":{
            "incremental_rendering":True,
            "whole_utterance_style_prompt":False,
            "random_humanization":False,
            "generic_tts_fallback":False,
            "fake_breath_audio":False,
            "speaker_profile_constant":True,
            "post_render_boundary_matching":True,
            "commit_only_delivered_audio":True,
        },
        "proof_ceiling":"The plan couples represented conversational state to bounded acoustic controls; perceptual authenticity still requires listening evidence.",
    }
    plan["plan_hash"]=digest(plan)
    return plan


def verify_realization_plan(plan: Dict[str,Any]) -> bool:
    if plan.get("schema")!=SCHEMA or plan.get("state_id")!=STATE_ID:
        raise ValueError("realization plan schema mismatch")
    bare={k:v for k,v in plan.items() if k!="plan_hash"}
    if digest(bare)!=plan.get("plan_hash"):
        raise ValueError("realization plan hash mismatch")
    identity=plan.get("identity",{})
    if identity.get("anchor_sha256")!=ANCHOR_SHA256 or identity.get("profile_sha256")!=PROFILE_SHA256:
        raise ValueError("acoustic identity mismatch")
    if identity.get("identity_changed") is not False:
        raise ValueError("identity change is prohibited")
    if not plan.get("units"):
        raise ValueError("no realization units")
    last=-1
    for u in plan["units"]:
        if u["index"]<=last: raise ValueError("nonmonotonic unit order")
        last=u["index"]
        if u["thought"] not in THOUGHT_CONTROLS: raise ValueError("unsupported thought")
        if not u["text"].strip(): raise ValueError("empty unit")
        if not .90<=u["controls"]["rate"]<=1.07: raise ValueError("rate outside bound")
        if not .92<=u["controls"]["volume"]<=1.06: raise ValueError("volume outside bound")
    return True
