"""Mari Unperformed Conversation v1.

This layer is deliberately subtractive. It preserves Mari's verified one-session
native continuity route while removing explicit performance actuation derived
from categorical thought/relationship state.

Internal state may change wording, structure, conversational action, or future
state. It does not automatically receive an acoustic gesture.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Dict, List

import numpy as np

from ..native_conversational_continuity_v1.planner import (
    PROFILE_SHA256,
    compile_native_continuity_plan,
    digest as native_digest,
    verify_native_continuity_plan,
)

STATE_ID="MARI_UNPERFORMED_CONVERSATION_v1"
SCHEMA="mari-unperformed-conversation-plan/1.0"

_CORRECTION=re.compile(r"^\s*(?:wait\b|actually\b|no[, ]+i mean\b|i mean\b)",re.I)

DISABLED_PERFORMANCE_CONTROLS=[
    "categorical_thought_trajectory",
    "categorical_thought_onset",
    "thought_rate_recipe",
    "thought_pause_recipe",
    "manual_semantic_prominence",
    "manual_phrase_finality",
    "relationship_volume_projection",
    "relationship_performance_projection",
    "random_disfluency",
    "fake_breath_audio",
    "prose_acting_instruction",
]

def _digest(value: Any) -> str:
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()

def _validate_conversation_plan(plan: Dict[str,Any]) -> None:
    if plan.get("schema")!="mari-conversational-plan/1.0":
        raise ValueError("unsupported conversational plan schema")
    if not plan.get("response_text","").strip():
        raise ValueError("response text required")
    if not plan.get("units"):
        raise ValueError("thought units required")
    contract=plan.get("renderer_contract",{})
    if contract.get("acoustic_identity_change") is not False:
        raise ValueError("acoustic identity mutation forbidden")
    if contract.get("generic_tts_fallback") is not False:
        raise ValueError("generic TTS fallback forbidden")
    identity=plan.get("state_before",{}).get("identity",{})
    if identity.get("profile_sha256")!=PROFILE_SHA256:
        raise ValueError("conversation plan does not target selected Mari profile")

def _action_state(plan: Dict[str,Any]) -> Dict[str,Any]:
    intent=copy.deepcopy(plan.get("intent",{}))
    listener=copy.deepcopy(plan.get("state_observed",{}).get("listener_model",{}))
    return {
        "action":intent.get("tactic","share"),
        "objective":intent.get("objective"),
        "target":intent.get("target"),
        "certainty":intent.get("certainty"),
        "listener_feedback":listener.get("last_feedback"),
        "listener_confusion":listener.get("confusion"),
        "listener_worry":listener.get("worry"),
        "listener_urgency":listener.get("urgency"),
        "relationship_is_context_not_performance_instruction":True,
    }

def _gate_units(plan: Dict[str,Any]) -> List[Dict[str,Any]]:
    events=[]
    for unit in plan["units"]:
        text=unit["text"]
        speech_act=unit.get("speech_act","assert")
        micro=[
            x.get("kind") for x in unit.get("microbehavior",[])
            if isinstance(x,dict) and isinstance(x.get("kind"),str)
        ]
        structural = speech_act=="repair" or bool(_CORRECTION.search(text)) or "repair_reset" in micro
        if structural:
            consequence="LEXICAL_STRUCTURAL"
            admitted=["spoken_words","clause_structure","native_uninstructed_delivery"]
            reason="the utterance itself contains a correction/revision; preserve the event without adding a performed correction gesture"
        elif speech_act=="ask":
            consequence="LEXICAL"
            admitted=["spoken_words","question_structure","native_uninstructed_delivery"]
            reason="questionhood is present in the utterance; no extra question-performance control is injected"
        else:
            consequence="INTERNAL_ONLY"
            admitted=["spoken_words","native_uninstructed_delivery"]
            reason="represented internal state does not independently justify an acoustic gesture"
        events.append({
            "unit_index":unit["index"],
            "text":text,
            "source_thought":unit.get("thought"),
            "speech_act":speech_act,
            "source_microbehavior":micro,
            "gate":consequence,
            "admitted":admitted,
            "injected_acoustic_control":False,
            "reason":reason,
        })
    return events

def _neutralize_native(native: Dict[str,Any]) -> Dict[str,Any]:
    out=copy.deepcopy(native)
    weights=np.asarray(out["trajectory"]["weights"],dtype=np.float32)
    out["trajectory"]["weights"]=np.zeros_like(weights).tolist()
    out["trajectory"]["onset"]=[0.0 for _ in out["trajectory"]["onset"]]
    for seg in out["trajectory"]["segments"]:
        seg["source_weight"]=seg.get("weight")
        seg["weight"]=0.0
        seg["interpretation"]="suppressed by MARI_UNPERFORMED_CONVERSATION_v1 audibility gate; event remains represented but receives no categorical acoustic gesture"
    out["trajectory"]["mode"]="UNPERFORMED_ZERO_INJECTED_ACTUATION"
    out["trajectory"]["all_injected_weights_zero"]=True
    out["laws"]["categorical_thought_prosody"]=False
    out["laws"]["categorical_relationship_prosody"]=False
    out["laws"]["manual_performance_projection"]=False
    out["proof_ceiling"]="One-session Mari continuity remains executable with zero injected categorical performance actuation. Conversationality remains a listening claim."
    bare={k:v for k,v in out.items() if k!="plan_hash"}
    out["plan_hash"]=native_digest(bare)
    verify_native_continuity_plan(out)
    return out

def compile_unperformed_conversation(
    conversation_plan: Dict[str,Any],
    tokenizer: Any,
    *,
    seed: int=140031,
) -> Dict[str,Any]:
    _validate_conversation_plan(conversation_plan)
    native=compile_native_continuity_plan(conversation_plan,tokenizer,seed=seed)
    native=_neutralize_native(native)
    events=_gate_units(conversation_plan)
    counts={}
    for event in events:
        counts[event["gate"]]=counts.get(event["gate"],0)+1
    plan={
        "schema":SCHEMA,
        "state_id":STATE_ID,
        "source_conversation_plan_hash":conversation_plan["plan_hash"],
        "identity":{
            "subject":"Mari404",
            "profile_sha256":PROFILE_SHA256,
            "identity_changed":False,
        },
        "action_state":_action_state(conversation_plan),
        "audibility_gate":{
            "events":events,
            "counts":counts,
            "rule":"internal state is acoustically silent unless a concrete conversational event requires more than wording/structure/native delivery",
            "injected_acoustic_controls":False,
        },
        "disabled_performance_controls":list(DISABLED_PERFORMANCE_CONTROLS),
        "native_plan":native,
        "laws":{
            "same_selected_Mari_profile":True,
            "one_decoder_session":True,
            "one_model_invocation":True,
            "waveform_stitching":False,
            "source_release_holds":False,
            "categorical_state_to_prosody":False,
            "relationship_to_performance":False,
            "native_uninstructed_delivery_primary":True,
            "lexical_revision_preserved":True,
        },
        "proof_ceiling":"This proves a subtractive executable path, not that it sounds conversational. Human listening determines whether removing performance direction improves Mari.",
    }
    plan["plan_hash"]=_digest(plan)
    verify_unperformed_plan(plan)
    return plan

def verify_unperformed_plan(plan: Dict[str,Any]) -> bool:
    if plan.get("schema")!=SCHEMA or plan.get("state_id")!=STATE_ID:
        raise ValueError("unperformed plan schema mismatch")
    supplied=plan.get("plan_hash")
    bare={k:v for k,v in plan.items() if k!="plan_hash"}
    if supplied!=_digest(bare):
        raise ValueError("unperformed plan hash mismatch")
    if plan["identity"]["profile_sha256"]!=PROFILE_SHA256 or plan["identity"]["identity_changed"] is not False:
        raise ValueError("Mari identity changed")
    native=plan["native_plan"]
    verify_native_continuity_plan(native)
    weights=np.asarray(native["trajectory"]["weights"],dtype=np.float32)
    onset=np.asarray(native["trajectory"]["onset"],dtype=np.float32)
    if np.count_nonzero(weights)!=0 or np.count_nonzero(onset)!=0:
        raise ValueError("explicit performance actuation is not zero")
    if any(h.get("applied_frames")!=0 for h in native["release"]["holds"]):
        raise ValueError("source withholding re-entered unperformed path")
    if plan["audibility_gate"]["injected_acoustic_controls"] is not False:
        raise ValueError("audibility gate injected performance control")
    if native["native_execution"]["decoder_sessions"]!=1 or native["native_execution"]["model_invocations"]!=1:
        raise ValueError("native continuity invariant failed")
    return True
