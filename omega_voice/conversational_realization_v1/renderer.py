"""Incremental waveform realization for MARI_CONVERSATIONAL_REALIZATION_LAYER_v1.

Each thought unit is synthesized with the same frozen Mari Voice v1 x-vector.
Only bounded rate, volume, timing and boundary operations are applied. There is
no prose style instruction, random disfluency, synthetic breath, or fallback
speaker.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf

from .layer import (
    ANCHOR_SHA256,
    PROFILE_SHA256,
    RECEIPT_SCHEMA,
    digest,
    verify_realization_plan,
)

EXPECTED_SR=24000


def sha256(path) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_wav(path) -> Dict[str,Any]:
    y,sr=sf.read(path,dtype="float32",always_2d=True)
    if sr!=EXPECTED_SR or y.shape[1]!=1 or len(y)<1200 or not np.isfinite(y).all():
        raise RuntimeError("malformed Mari render")
    mono=y[:,0]
    rms=float(np.sqrt(np.mean(mono**2)))
    peak=float(np.max(np.abs(mono)))
    if rms<1e-5 or peak>=1.0:
        raise RuntimeError("silent or clipped Mari render")
    return {
        "sample_rate_hz":sr,
        "frames":len(mono),
        "duration_s":len(mono)/sr,
        "rms":rms,
        "peak":peak,
        "sha256":sha256(path),
    }


def _window_rms(y: np.ndarray, n: int) -> float:
    if not len(y): return 0.0
    z=y[:n] if n>0 else y
    return float(np.sqrt(np.mean(z**2))) if len(z) else 0.0


def _tail_rms(y: np.ndarray, n: int) -> float:
    if not len(y): return 0.0
    z=y[-n:] if n>0 else y
    return float(np.sqrt(np.mean(z**2))) if len(z) else 0.0


def _active_bounds(y: np.ndarray, sr: int) -> tuple[int,int]:
    # Keep natural model silence, but reject pathological leading/trailing dead air.
    win=max(48,int(sr*.01))
    if len(y)<win*4:return 0,len(y)
    energy=np.convolve(y*y,np.ones(win,dtype=np.float32)/win,mode="same")
    threshold=max(1e-6,float(np.max(energy))*.0025)
    active=np.flatnonzero(energy>threshold)
    if not len(active):return 0,len(y)
    pad=int(sr*.025)
    return max(0,int(active[0])-pad),min(len(y),int(active[-1])+pad)


def _bounded_boundary_gain(previous: np.ndarray, current: np.ndarray, sr: int) -> tuple[np.ndarray,float]:
    n=max(1,int(sr*.08))
    a=_tail_rms(previous,n)
    b=_window_rms(current,n)
    if a<1e-5 or b<1e-5:return current,1.0
    ratio=math.sqrt(a/b)
    gain=float(np.clip(ratio,.97,1.03))
    return (current*gain).astype(np.float32),gain


def _join(previous: np.ndarray, current: np.ndarray, sr: int, pause_s: float, mode: str) -> tuple[np.ndarray,Dict[str,Any]]:
    current,gain=_bounded_boundary_gain(previous,current,sr)
    crossfade_s=.0
    # Flow/commit joins preserve continuity; reflective and repair boundaries stay audible.
    if mode in {"flow","commit"} and pause_s<=.065:
        n=min(int(sr*.028),len(previous)//4,len(current)//4)
        if n>16:
            ramp=np.linspace(0,1,n,dtype=np.float32)
            blend=previous[-n:]*(1-ramp)+current[:n]*ramp
            joined=np.concatenate([previous[:-n],blend,current[n:]])
            crossfade_s=n/sr
            return joined,{"boundary_gain":gain,"crossfade_s":crossfade_s,"inserted_silence_s":0.0}
    silence=np.zeros(int(round(sr*pause_s)),dtype=np.float32)
    joined=np.concatenate([previous,silence,current])
    return joined,{"boundary_gain":gain,"crossfade_s":0.0,"inserted_silence_s":len(silence)/sr}


def _render_engine_unit(
    *,
    engine: str,
    model: str,
    profile: str,
    text: str,
    output: str,
    seed: int,
    rate: float,
    volume: float,
    timeout: int=480,
) -> Dict[str,Any]:
    for p in (engine,profile):
        if not Path(p).is_file(): raise FileNotFoundError(p)
    if not Path(model).is_dir(): raise FileNotFoundError(model)
    cmd=[
        engine,"-d",model,"--load-voice",profile,"--xvector-only",
        "-l","English","--text",text,
        "--seed",str(seed),"--temperature","0.42","--top-k","40","--top-p","0.95",
        "--rep-penalty","1.05","--rate",f"{rate:.4f}","--volume",f"{volume:.4f}",
        "--onset-fade","8","--tail-trim","-j4","-o",output,
    ]
    cp=subprocess.run(
        cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout,
        env={**os.environ,"OPENBLAS_NUM_THREADS":"4"},
    )
    if cp.returncode!=0 or not Path(output).is_file():
        raise RuntimeError("Mari unit renderer failed; no fallback used\n"+cp.stdout[-3000:])
    return {"command":cmd,"log":cp.stdout,"audio":inspect_wav(output)}


def render_incremental(
    realization_plan: Dict[str,Any],
    *,
    engine: str,
    model: str,
    profile: str,
    output: str,
    unit_directory: Optional[str]=None,
) -> Dict[str,Any]:
    verify_realization_plan(realization_plan)
    if sha256(profile)!=PROFILE_SHA256:
        raise RuntimeError("Mari Voice v1 profile hash mismatch")
    out=Path(output)
    if out.exists(): raise FileExistsError(output)
    out.parent.mkdir(parents=True,exist_ok=True)

    receipts=[]
    combined=None
    sr=EXPECTED_SR
    base_dir=Path(unit_directory) if unit_directory else out.parent/(out.stem+"_units")
    base_dir.mkdir(parents=True,exist_ok=True)

    for pos,unit in enumerate(realization_plan["units"]):
        unit_path=base_dir/f"unit_{unit['index']:02d}.wav"
        if unit_path.exists(): unit_path.unlink()
        controls=unit["controls"]
        raw=_render_engine_unit(
            engine=engine,model=model,profile=profile,text=unit["text"],output=str(unit_path),
            seed=unit["seed"],rate=controls["rate"],volume=controls["volume"],
        )
        y,r=sf.read(unit_path,dtype="float32",always_2d=True)
        if r!=sr: raise RuntimeError("sample-rate drift across units")
        mono=y[:,0]
        lo,hi=_active_bounds(mono,sr)
        mono=mono[lo:hi]
        boundary={"boundary_gain":1.0,"crossfade_s":0.0,"inserted_silence_s":0.0}
        if combined is None:
            latency=float(realization_plan["continuity"]["initial_response_latency_s"])
            combined=np.concatenate([np.zeros(int(round(sr*latency)),dtype=np.float32),mono])
            boundary["inserted_silence_s"]=latency
        else:
            combined,boundary=_join(combined,mono,sr,float(controls["pre_pause_s"]),controls["join"])
        receipts.append({
            "index":unit["index"],
            "text":unit["text"],
            "thought":unit["thought"],
            "speech_act":unit["speech_act"],
            "seed":unit["seed"],
            "controls":controls,
            "continuity_token":unit["continuity_token"],
            "raw_audio":raw["audio"],
            "trimmed_frames":len(mono),
            "boundary":boundary,
            "self_monitor_hook":{
                "delivered_unit_index":unit["index"],
                "delivered_text_hash":digest(unit["text"]),
                "eligible_for_next_unit_context":True,
            },
        })

    if combined is None or not len(combined): raise RuntimeError("no rendered units")
    peak=float(np.max(np.abs(combined)))
    if peak>.985:
        combined=combined*(.985/peak)
    sf.write(out,combined,sr,subtype="PCM_16")
    final=inspect_wav(out)
    receipt={
        "schema":RECEIPT_SCHEMA,
        "state_id":realization_plan["state_id"],
        "status":"INCREMENTAL_RENDER_COMPLETE",
        "plan_hash":realization_plan["plan_hash"],
        "identity":{
            "subject":"Mari404",
            "anchor_sha256":ANCHOR_SHA256,
            "profile_sha256":PROFILE_SHA256,
            "identity_changed":False,
        },
        "output":final,
        "units":receipts,
        "continuity":{
            "terminal_token":realization_plan["continuity"]["terminal_token"],
            "terminal_breath_reserve":realization_plan["continuity"]["terminal_breath_reserve"],
            "speaker_profile_constant":True,
            "unit_count":len(receipts),
        },
        "render_policy":{
            "generic_tts_fallback":False,
            "prose_style_instruction":False,
            "random_disfluency":False,
            "synthetic_breath_audio":False,
            "bounded_rate_volume_controls":True,
            "measured_boundary_matching":True,
        },
        "proof_ceiling":"Successful rendering and continuity mechanics do not by themselves prove perceptual conversational authenticity.",
    }
    return receipt


def render_whole_utterance_baseline(
    realization_plan: Dict[str,Any],
    *,
    engine: str,
    model: str,
    profile: str,
    output: str,
    seed: int=120031,
) -> Dict[str,Any]:
    """Matched-text baseline: one neutral whole-utterance render, same speaker profile."""
    verify_realization_plan(realization_plan)
    text=" ".join(u["text"] for u in realization_plan["units"])
    result=_render_engine_unit(
        engine=engine,model=model,profile=profile,text=text,output=output,
        seed=seed,rate=1.0,volume=1.0,
    )
    return {
        "schema":"mari-conversational-realization-baseline/1.0",
        "plan_hash":realization_plan["plan_hash"],
        "text":text,
        "output":result["audio"],
        "identity":{"anchor_sha256":ANCHOR_SHA256,"profile_sha256":PROFILE_SHA256},
        "whole_utterance":True,
        "generic_tts_fallback":False,
    }
