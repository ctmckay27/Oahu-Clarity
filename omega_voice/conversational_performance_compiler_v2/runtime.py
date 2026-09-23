"""Mari Conversational Performance Compiler v2.

CPC v2 preserves the verified v1 discourse/finality path and adds only the new
held-out-qualified native tempo axis. Onset pressure and local focus remain
explicitly rejected and cannot enter this runtime.
"""
from __future__ import annotations

import hashlib,json
from pathlib import Path
from typing import Any,Dict
import numpy as np

from omega_voice.conversational_performance_compiler_v1 import compile_native_performance, verify_native_performance_plan
from omega_voice.generative_v5.native import render as native_render, write_trajectory
from omega_voice.generative_v5.text_release import write_release
from omega_voice.native_prosody_basis_v1.runtime import AXES,QUALIFIED,basis_sha256,load_basis

STATE_ID="MARI_CONVERSATIONAL_PERFORMANCE_COMPILER_v2"
SCHEMA="mari-conversational-performance-v2/1.0"
RECEIPT_SCHEMA="mari-conversational-performance-v2-receipt/1.0"
TEMPO_LIMIT=.18

def _digest(v:Any)->str:
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def _tempo_weights(source:Dict[str,Any],frame_count:int)->np.ndarray:
    samples=np.asarray(source["performance_plan"]["performance_space"]["channels"]["tempo_tendency"],dtype=np.float32)
    if samples.ndim!=1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("invalid CPC v1 tempo intention")
    samples=np.clip(samples,-TEMPO_LIMIT,TEMPO_LIMIT)
    release=source["native_plan"]["release"]["earliest_frames"]
    active_end=min(frame_count,max(24,int(release[-1])+48))
    out=np.zeros(frame_count,dtype=np.float32)
    if active_end:
        out[:active_end]=np.interp(
            np.linspace(0,1,active_end,dtype=np.float32),
            np.linspace(0,1,len(samples),dtype=np.float32),
            samples,
        )
    return out

def compile_native_performance_v2(conversation_plan:Dict[str,Any],tokenizer:Any,*,seed:int=180001)->Dict[str,Any]:
    source=compile_native_performance(conversation_plan,tokenizer,seed=seed)
    verify_native_performance_plan(source)
    native=source["native_plan"]
    finality=np.asarray(native["trajectory"]["weights"],dtype=np.float32)
    if finality.ndim!=1: raise ValueError("unexpected CPC v1 finality shape")
    tempo=_tempo_weights(source,len(finality))
    weights=np.stack([finality,tempo],axis=1)
    plan={
      "schema":SCHEMA,"state_id":STATE_ID,
      "source_cpc_v1_plan_hash":source["plan_hash"],
      "source_conversation_plan_hash":conversation_plan["plan_hash"],
      "text":native["text"],"seed":native["seed"],
      "identity":dict(native["identity"]),
      "basis":{"axes":list(AXES),"sha256":basis_sha256(),"qualified":dict(QUALIFIED)},
      "trajectory":{
        "frame_count":len(weights),"weights":weights.tolist(),"onset":[0.0,0.0],
        "finality_source":"CPC_v1 discourse-derived measured same-speaker axis",
        "tempo_source":"CPC_v1 continuous tempo_tendency projected through held-out-qualified same-Mari tempo axis",
        "tempo_limit":TEMPO_LIMIT,
        "rejected_axes_not_present":["onset_pressure","local_focus"],
      },
      "source_native_plan":native,
      "source_performance_plan":source["performance_plan"],
      "execution_contract":{
        "decoder_sessions":1,"model_invocations":1,"unit_restarts":0,
        "codec_state_resets":0,"kv_cache_resets":0,"waveform_stitching":False,
        "source_release_holds":False,"raw_cross_speaker_reference":False,
        "generic_tts_fallback":False,"fake_breath_audio":False,
      },
      "proof_ceiling":"CPC v2 actuates only qualified finality/continuation and tempo. It does not claim onset pressure, local focus, respiration, or perceptual naturalness are solved."
    }
    plan["plan_hash"]=_digest(plan)
    verify_plan_v2(plan)
    return plan

def verify_plan_v2(plan:Dict[str,Any])->bool:
    if plan.get("schema")!=SCHEMA or plan.get("state_id")!=STATE_ID: raise ValueError("CPC v2 schema mismatch")
    supplied=plan.get("plan_hash");bare={k:v for k,v in plan.items() if k!="plan_hash"}
    if supplied!=_digest(bare): raise ValueError("CPC v2 hash mismatch")
    if plan["basis"]["qualified"].get("tempo") is not True: raise ValueError("tempo basis not qualified")
    if plan["basis"]["qualified"].get("onset_pressure") is not False or plan["basis"]["qualified"].get("local_focus") is not False: raise ValueError("rejected axes reopened")
    w=np.asarray(plan["trajectory"]["weights"],dtype=np.float32)
    if w.shape!=(plan["trajectory"]["frame_count"],2) or not np.isfinite(w).all(): raise ValueError("CPC v2 trajectory shape")
    if np.max(np.abs(w[:,0]))>.45 or np.max(np.abs(w[:,1]))>TEMPO_LIMIT+1e-6: raise ValueError("CPC v2 trajectory envelope")
    if any(h.get("applied_frames")!=0 for h in plan["source_native_plan"]["release"]["holds"]): raise ValueError("source starvation re-entered")
    return True

def materialize_packets_v2(plan:Dict[str,Any],directory:str|Path)->Dict[str,Any]:
    verify_plan_v2(plan)
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    n=plan["source_native_plan"]
    rel=write_release(d/"cpc_v2.mrl",n["tokenizer"]["token_ids"],n["release"]["earliest_frames"])
    traj=write_trajectory(d/"cpc_v2.mtraj",load_basis(),np.asarray(plan["trajectory"]["weights"],dtype=np.float32),onset=np.zeros(2,dtype=np.float32))
    return {"release":rel,"trajectory":traj}

def render_native_performance_v2(root:str|Path,plan:Dict[str,Any],output:str|Path)->Dict[str,Any]:
    verify_plan_v2(plan)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError(out)
    packets=materialize_packets_v2(plan,out.parent/(out.stem+"_cpc_v2_packets"))
    record=native_render(root,plan["text"],out,seed=plan["seed"],trajectory=packets["trajectory"]["path"],
                         incremental_text=True,text_release=packets["release"]["path"],profile_mode="xvector")
    if record.get("profile_file_sha256")!=plan["identity"]["profile_sha256"]: raise RuntimeError("Mari profile changed")
    if not record.get("text_consumption",{}).get("no_early_consumption"): raise RuntimeError("source release audit failed")
    if record.get("trajectory_sha256")!=packets["trajectory"]["sha256"]: raise RuntimeError("prosody basis trajectory receipt mismatch")
    weights=np.asarray(plan["trajectory"]["weights"])
    receipt={
      "schema":RECEIPT_SCHEMA,"state_id":STATE_ID,"status":"CPC_V2_NATIVE_RENDER_COMPLETE",
      "plan_hash":plan["plan_hash"],"output":record["audio"],
      "identity":{"profile_file_sha256":record["profile_file_sha256"],"identity_changed":False},
      "basis":plan["basis"],
      "tempo":{"max_abs_weight":float(np.max(np.abs(weights[:,1]))),"mean_weight":float(np.mean(weights[:,1]))},
      "native_execution":{"decoder_sessions":1,"model_invocations":1,"unit_restarts":0,"codec_state_resets":0,"kv_cache_resets":0,"waveform_stitching":False},
      "text_consumption":record["text_consumption"],
      "render_policy":{"source_release_holds":False,"raw_cross_speaker_reference":False,"prose_style_instruction":False,"generic_tts_fallback":False},
      "native_record":record,
      "proof_ceiling":"Execution proves the two-axis qualified native path ran. Perceptual conversational improvement remains a listening claim."
    }
    rp=out.with_suffix(".cpc-v2.receipt.json");rp.write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt
