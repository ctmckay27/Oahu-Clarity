"""Waveform realization for Mari Acoustic Actuator v2.

The pinned engine supplies G1-1 speech. Expanded controls are realized as
bounded deterministic post-render operations, so they have actual waveform
consequences without replacing the frozen speaker profile.
"""
from __future__ import annotations
import copy, json, math
from pathlib import Path
from typing import Any, Dict
import numpy as np
import soundfile as sf
from omega_voice.conversational_realization_v1.renderer import (
    EXPECTED_SR,_active_bounds,_join,_render_engine_unit,inspect_wav,sha256
)
from omega_voice.conversational_realization_v1.layer import PROFILE_SHA256

def _clamp(x,a,b): return max(a,min(b,float(x)))

def _apply_expanded(y:np.ndarray,sr:int,c:Dict[str,float]) -> tuple[np.ndarray,Dict[str,Any]]:
    z=y.astype(np.float32).copy()
    # onset energy: bounded gain over first 55 ms, tapering to unity
    n=min(len(z),max(1,int(sr*.055)))
    onset=float(c["onset_energy"]); onset_gain=1.0+(onset-.52)*1.35
    if n:
        env=np.linspace(onset_gain,1.0,n,dtype=np.float32); z[:n]*=env
    # consonant-energy proxy: high-frequency pre-emphasis, intentionally subtle
    ce=float(c["consonant_energy"]); alpha=(ce-.55)*.22
    if abs(alpha)>1e-6 and len(z)>1:
        d=np.empty_like(z); d[0]=0; d[1:]=z[1:]-z[:-1]; z=z+alpha*d
    # finality: shape final 140 ms. Higher finality gives a firmer release.
    nf=min(len(z),max(1,int(sr*.14))); finality=float(c["finality"])
    end_gain=1.0+(finality-.645)*.22
    if nf:
        z[-nf:]*=np.linspace(1.0,end_gain,nf,dtype=np.float32)
    # timing-flex: tiny deterministic local time deformation by interpolation.
    tf=float(c["timing_flex"]); factor=1.0+(tf-.10)*.12
    if len(z)>32 and abs(factor-1)>1e-4:
        x=np.arange(len(z),dtype=np.float64)
        nx=np.linspace(0,len(z)-1,max(32,int(round(len(z)*factor))),dtype=np.float64)
        z=np.interp(nx,x,z).astype(np.float32)
    peak=float(np.max(np.abs(z))) if len(z) else 0
    if peak>.985:z*=.985/peak
    return z,{"onset_gain":round(onset_gain,5),"preemphasis_alpha":round(alpha,5),
              "final_gain":round(end_gain,5),"time_factor":round(factor,5)}

def render_actuated(plan:Dict[str,Any],*,engine:str,model:str,profile:str,output:str,unit_directory:str|None=None)->Dict[str,Any]:
    if plan.get("schema")!="mari-acoustic-actuator/2.0":raise ValueError("bad actuator plan")
    if plan["identity"]["change"] is not False:raise ValueError("identity change forbidden")
    if sha256(profile)!=PROFILE_SHA256:raise RuntimeError("Mari profile hash mismatch")
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(output)
    base=Path(unit_directory) if unit_directory else out.parent/(out.stem+"_units");base.mkdir(parents=True,exist_ok=True)
    combined=None; receipts=[]
    for pos,u in enumerate(plan["units"]):
        rawp=base/f"raw_{u['index']:02d}.wav"
        if rawp.exists():rawp.unlink()
        raw=_render_engine_unit(engine=engine,model=model,profile=profile,text=u["text"],output=str(rawp),
             seed=u["seed"],rate=u["controls"]["rate"],volume=u["controls"]["volume"])
        y,sr=sf.read(rawp,dtype="float32",always_2d=True); mono=y[:,0]
        lo,hi=_active_bounds(mono,sr); mono=mono[lo:hi]
        mono,expanded=_apply_expanded(mono,sr,u["expanded_controls"])
        actp=base/f"actuated_{u['index']:02d}.wav"; sf.write(actp,mono,sr,subtype="PCM_16")
        if combined is None:
            pause=float(u["controls"]["pre_pause_s"]); combined=np.concatenate([np.zeros(int(sr*pause),np.float32),mono])
            boundary={"inserted_silence_s":pause,"crossfade_s":0.0,"boundary_gain":1.0}
        else:
            combined,boundary=_join(combined,mono,sr,float(u["controls"]["pre_pause_s"]),u["controls"]["join"])
        receipts.append({"index":u["index"],"text":u["text"],"raw":raw["audio"],"actuated":inspect_wav(actp),
                         "native_controls":u["controls"],"expanded_controls":u["expanded_controls"],
                         "realized_operations":expanded,"boundary":boundary})
    peak=float(np.max(np.abs(combined)))
    if peak>.985:combined*=.985/peak
    sf.write(out,combined,EXPECTED_SR,subtype="PCM_16")
    return {"schema":"mari-acoustic-actuator-render-receipt/2.0","actuator_hash":plan["actuator_hash"],
            "identity":{"profile_sha256":PROFILE_SHA256,"speaker_profile_constant":True},
            "output":inspect_wav(out),"units":receipts,
            "proof_ceiling":"Waveform actuation is verified mechanically; Mari identity and perceptual improvement require blinded listening."}

def patched_condition(plan:Dict[str,Any],patch:Dict[str,float])->Dict[str,Any]:
    out=copy.deepcopy(plan)
    for u in out["units"]:
        for k,v in patch.items():
            if k in {"rate","volume"}:u["controls"][k]=float(v)
            elif k in u["expanded_controls"]:u["expanded_controls"][k]=float(v)
            else:raise ValueError("unsupported sweep axis "+k)
    return out
