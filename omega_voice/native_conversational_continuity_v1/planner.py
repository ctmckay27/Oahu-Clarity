"""Compile Mari conversational state into one native decoder trajectory.

The final waveform is generated in one Qwen decoder session. Conversational
thought boundaries affect source-bound token availability and one measured
same-speaker boundary-contour direction. Unsupported acoustic dimensions remain
unrealized rather than being replaced with style prose or stitched clips.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .calibrated_finality import (
    COSINE_TO_SOURCE,
    RELATIVE_L2_ERROR,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_DIRECTION_RAW_SHA256,
    SOURCE_NPZ_SHA256,
    load_direction,
)
from ..generative_v5.native import write_trajectory
from ..generative_v5.text_release import write_release

STATE_ID = "MARI_NATIVE_CONVERSATIONAL_CONTINUITY_LAYER_v1"
SCHEMA = "mari-native-conversational-continuity-plan/1.0"
ANCHOR_SHA256 = "73bb894fc75e18dddf76e3327f3bf48e161b7dd8b30fbe4949c3ac7318c29567"
PROFILE_SHA256 = "9c49146ae2bd3c5a32c686ad1705fd912391af4a39db0d63e9dc353640d304fa"
FRAME_S = 0.08

# The sole native acoustic actuator is the measured finality/continuation axis.
# Signs and magnitudes below are bounded hypotheses, intentionally below the
# +/-0.7 held-out calibration envelope. They are not emotion presets.
THOUGHT_WEIGHT = {
    "known": 0.12,
    "observing": 0.04,
    "judging": 0.16,
    "searching": -0.22,
    "remembering": -0.08,
    "deciding": -0.12,
    "reconsidering": -0.20,
    "correcting": -0.30,
    "realizing": 0.24,
    "discovering": 0.18,
    "suppressing": -0.12,
    "withholding": -0.18,
    "changing_tactic": -0.10,
}

THOUGHT_HOLD_FRAMES = {
    "known": 0,
    "observing": 0,
    "judging": 0,
    "searching": 2,
    "remembering": 1,
    "deciding": 2,
    "reconsidering": 2,
    "correcting": 2,
    "realizing": 0,
    "discovering": 0,
    "suppressing": 1,
    "withholding": 1,
    "changing_tactic": 1,
}


def digest(value: Any) -> str:
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _validate_conversation_plan(plan: Dict[str,Any]) -> None:
    if plan.get("schema") != "mari-conversational-plan/1.0":
        raise ValueError("unsupported conversational plan schema")
    if not isinstance(plan.get("response_text"),str) or not plan["response_text"].strip():
        raise ValueError("conversation plan has no response text")
    if not isinstance(plan.get("units"),list) or not plan["units"]:
        raise ValueError("conversation plan has no thought units")
    identity=plan.get("state_before",{}).get("identity",{})
    if identity.get("anchor_sha256")!=ANCHOR_SHA256 or identity.get("profile_sha256")!=PROFILE_SHA256:
        raise ValueError("conversation plan targets another acoustic identity")
    contract=plan.get("renderer_contract",{})
    if contract.get("acoustic_identity_change") is not False:
        raise ValueError("identity change is prohibited")
    if contract.get("generic_tts_fallback") is not False:
        raise ValueError("generic TTS fallback must be disabled")
    if contract.get("random_disfluency") is not False:
        raise ValueError("random disfluency is prohibited")


def _locate_units(text: str, units: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    cursor=0;located=[]
    for unit in units:
        phrase=unit.get("text","")
        if not phrase.strip(): raise ValueError("empty conversational unit")
        start=text.find(phrase,cursor)
        if start<0: raise ValueError("unit text is not an ordered substring of response text")
        end=start+len(phrase)
        located.append({**copy.deepcopy(unit),"char_start":start,"char_end":end})
        cursor=end
    return located


def _tokenize(tokenizer: Any, text: str) -> Tuple[List[int],List[Tuple[int,int]]]:
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=list(encoded["input_ids"]);offsets=[tuple(x) for x in encoded["offset_mapping"]]
    if not ids or len(ids)!=len(offsets): raise ValueError("tokenizer did not return aligned token offsets")
    if any(len(x)!=2 or x[0]<0 or x[1]<x[0] for x in offsets): raise ValueError("invalid token offsets")
    return ids,offsets


def _assign_tokens(located: List[Dict[str,Any]], offsets: List[Tuple[int,int]]) -> List[Dict[str,Any]]:
    assigned=[]
    for unit in located:
        hits=[i for i,(a,b) in enumerate(offsets) if b>unit["char_start"] and a<unit["char_end"]]
        if not hits: raise ValueError("thought unit has no source tokens")
        if hits!=list(range(hits[0],hits[-1]+1)): raise ValueError("noncontiguous unit token span")
        assigned.append({**unit,"token_start":hits[0],"token_end":hits[-1]+1})
    if assigned[0]["token_start"]!=0 or assigned[-1]["token_end"]!=len(offsets):
        raise ValueError("thought units do not cover the complete token sequence")
    for a,b in zip(assigned,assigned[1:]):
        if a["token_end"]!=b["token_start"]: raise ValueError("gap or overlap between thought-unit tokens")
    return assigned


def _compile_release_frames(units: List[Dict[str,Any]], token_count: int) -> Tuple[List[int],List[Dict[str,Any]]]:
    # Evidence-bounded repair, 2026-09-22:
    # delaying lexical source tokens at represented thought boundaries can starve
    # the live decoder late in an utterance and provoke premature EOS. Production
    # probe evidence showed the same frozen Mari profile/seed/engine improved from
    # WER 0.285714 to 0.020408 when these extra holds were removed, while one-session
    # continuity and the measured acoustic trajectory remained active.
    #
    # Source availability therefore advances monotonically at one token per native
    # codec frame with no additional semantic/respiratory withholding. Thought
    # state remains causally represented through the native acoustic trajectory.
    # Physical quiet-intake timing is recorded below as an unapplied request until
    # a non-starving native pause mechanism is qualified.
    frames=[0]+list(range(token_count))
    holds=[]
    for unit in units:
        thought=unit.get("thought")
        if thought not in THOUGHT_HOLD_FRAMES: raise ValueError("unsupported thought state: "+str(thought))
        quiet=float(unit.get("quiet_intake_before_s",0.0) or 0.0)
        if not math.isfinite(quiet) or quiet<0 or quiet>1.5: raise ValueError("invalid physical quiet-intake duration")
        requested=max(THOUGHT_HOLD_FRAMES[thought],int(math.ceil(quiet/FRAME_S)))
        holds.append({
            "unit_index":unit["index"],"thought":thought,"token_start":unit["token_start"],
            "requested_frames":requested,"applied_frames":0,"cumulative_frames":0,
            "cause":"legacy represented thought boundary and/or physical quiet-intake request",
            "status":"RECORDED_NOT_APPLIED",
            "reason":"source-token holds caused verified late-utterance starvation; timing must use a non-starving actuator",
        })
    for a,b in zip(frames,frames[1:]):
        if b<a: raise RuntimeError("release chronology became nonmonotonic")
    if frames[0]!=0 or frames[-1]>8190: raise ValueError("release schedule outside native envelope")
    return frames,holds


def _compile_trajectory(units: List[Dict[str,Any]], release_frames: List[int]) -> Tuple[np.ndarray,np.ndarray,List[Dict[str,Any]]]:
    last_release=release_frames[-1]
    frame_count=min(8192,max(256,last_release+128))
    weights=np.zeros((frame_count,1),dtype=np.float32)
    segments=[]
    for pos,unit in enumerate(units):
        thought=unit["thought"]
        if thought not in THOUGHT_WEIGHT: raise ValueError("unsupported thought state: "+str(thought))
        weight=float(THOUGHT_WEIGHT[thought])
        source_start=release_frames[unit["token_start"]]
        if pos+1<len(units):
            source_end=release_frames[units[pos+1]["token_start"]]
        else:
            source_end=min(frame_count-1,last_release+36)
        # Act locally around the represented unit boundary. This preserves the
        # measured scope of the direction instead of painting an entire sentence.
        first=max(source_start,source_end-8)
        last=min(frame_count,max(first+1,source_end+3))
        weights[first:last,0]=weight
        segments.append({
            "unit_index":unit["index"],"thought":thought,"weight":weight,
            "source_token_span":[unit["token_start"],unit["token_end"]],
            "native_frame_window":[first,last],
            "interpretation":"measured local boundary-contour actuation only; not an emotion or personality label",
        })
    initial=float(THOUGHT_WEIGHT[units[0]["thought"]])
    onset=np.asarray([float(np.clip(initial*.35,-.12,.12))],dtype=np.float32)
    return weights,onset,segments


def compile_native_continuity_plan(
    conversation_plan: Dict[str,Any],
    tokenizer: Any,
    *,
    seed: int=130021,
) -> Dict[str,Any]:
    _validate_conversation_plan(conversation_plan)
    if not isinstance(seed,int) or seed<0: raise ValueError("seed must be a nonnegative integer")
    text=conversation_plan["response_text"]
    units=_assign_tokens(_locate_units(text,conversation_plan["units"]),_tokenize(tokenizer,text)[1])
    token_ids,offsets=_tokenize(tokenizer,text)
    # Reassignment with the exact returned offsets guards tokenization mutation.
    units=_assign_tokens(_locate_units(text,conversation_plan["units"]),offsets)
    release_frames,holds=_compile_release_frames(units,len(token_ids))
    weights,onset,segments=_compile_trajectory(units,release_frames)
    direction=load_direction()
    plan={
        "schema":SCHEMA,
        "state_id":STATE_ID,
        "conversation_plan_hash":conversation_plan["plan_hash"],
        "session_id":conversation_plan["session_id"],
        "turn":conversation_plan["turn"],
        "listener_id":conversation_plan["listener_id"],
        "text":text,
        "seed":seed,
        "identity":{
            "subject":"Mari404","anchor_sha256":ANCHOR_SHA256,
            "profile_sha256":PROFILE_SHA256,"identity_changed":False,
        },
        "native_execution":{
            "decoder_sessions":1,"model_invocations":1,"unit_restarts":0,
            "codec_state_resets":0,"kv_cache_resets":0,"profile_mode":"xvector",
            "incremental_text":True,"prose_instructions":False,"generic_fallback":False,
        },
        "tokenizer":{
            "token_count":len(token_ids),"token_ids":token_ids,
            "offset_mapping":[list(x) for x in offsets],
        },
        "units":[{
            "index":u["index"],"text":u["text"],"thought":u["thought"],
            "speech_act":u.get("speech_act"),"certainty":u.get("certainty"),
            "token_span":[u["token_start"],u["token_end"]],
            "quiet_intake_before_s":u.get("quiet_intake_before_s",0.0),
        } for u in units],
        "release":{"earliest_frames":release_frames,"holds":holds,"frame_period_s":FRAME_S,
                   "policy":"NO_EXTRA_SOURCE_WITHHOLDING_AFTER_STARVATION_EVIDENCE",
                   "legacy_hold_requests_recorded_not_applied":True},
        "trajectory":{
            "frame_count":len(weights),"weights":weights[:,0].tolist(),
            "onset":onset.tolist(),"segments":segments,
            "bank_shape":list(direction.shape),
            "source_archive_sha256":SOURCE_ARCHIVE_SHA256,
            "source_npz_sha256":SOURCE_NPZ_SHA256,
            "source_direction_raw_sha256":SOURCE_DIRECTION_RAW_SHA256,
            "compact_cosine_to_source":COSINE_TO_SOURCE,
            "compact_relative_l2_error":RELATIVE_L2_ERROR,
        },
        "laws":{
            "one_decoder_session":True,"one_continuous_codec_stream":True,
            "same_speaker_profile":True,"separate_unit_synthesis":False,
            "waveform_stitching":False,"style_prompting":False,
            "random_humanization":False,"fake_breath_audio":False,
            "unsupported_dimensions_remain_unrealized":True,
            "source_release_holds":False,
            "physical_quiet_intake_via_source_withholding":False,
        },
        "proof_ceiling":"Native continuity and measured trajectory actuation are executable. Source-token holds are disabled after starvation evidence; physical quiet-intake timing remains unresolved in this route. Person-like authenticity remains a listening claim.",
    }
    plan["plan_hash"]=digest(plan)
    return plan


def verify_native_continuity_plan(plan: Dict[str,Any]) -> bool:
    if plan.get("schema")!=SCHEMA or plan.get("state_id")!=STATE_ID:
        raise ValueError("native continuity plan schema mismatch")
    bare={k:v for k,v in plan.items() if k!="plan_hash"}
    if digest(bare)!=plan.get("plan_hash"): raise ValueError("native continuity plan hash mismatch")
    if plan["identity"]["anchor_sha256"]!=ANCHOR_SHA256 or plan["identity"]["profile_sha256"]!=PROFILE_SHA256:
        raise ValueError("Mari identity mismatch")
    native=plan["native_execution"]
    required={"decoder_sessions":1,"model_invocations":1,"unit_restarts":0,"codec_state_resets":0,"kv_cache_resets":0}
    for key,value in required.items():
        if native.get(key)!=value: raise ValueError("native continuity invariant failed: "+key)
    frames=plan["release"]["earliest_frames"]
    if frames[0]!=0 or len(frames)!=plan["tokenizer"]["token_count"]+1:
        raise ValueError("release packet shape mismatch")
    if any(a>b for a,b in zip(frames,frames[1:])): raise ValueError("nonmonotonic release schedule")
    weights=np.asarray(plan["trajectory"]["weights"],dtype=np.float32)
    if not len(weights) or not np.isfinite(weights).all() or np.max(np.abs(weights))>.7:
        raise ValueError("trajectory outside calibrated envelope")
    return True


def materialize_packets(plan: Dict[str,Any], directory: str|Path) -> Dict[str,Any]:
    verify_native_continuity_plan(plan)
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    release_path=d/"native_continuity.mrl"
    trajectory_path=d/"native_continuity.mtraj"
    release=write_release(release_path,plan["tokenizer"]["token_ids"],plan["release"]["earliest_frames"])
    bank=load_direction()
    weights=np.asarray(plan["trajectory"]["weights"],dtype=np.float32)[:,None]
    onset=np.asarray(plan["trajectory"]["onset"],dtype=np.float32)
    trajectory=write_trajectory(trajectory_path,bank,weights,onset=onset)
    plan_path=d/"native_continuity_plan.json"
    plan_path.write_text(json.dumps(plan,indent=2)+"\n")
    return {"release":release,"trajectory":trajectory,"plan_path":str(plan_path),"plan_sha256":hashlib.sha256(plan_path.read_bytes()).hexdigest()}
