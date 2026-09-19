"""Frozen carrier realization and causal temporal adapter.

The production gate exposes unsupported acts instead of silently discarding them.
Timing and level actuators are signal-domain operations, not claimed physiological
simulation. Full Mari voice completion cannot be inferred from these actuators.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from .runtime import ANCHOR, PROFILE, verify_plan, digest

ENGINE_REV="e391ec5467b0218eeb175f4888ad65b259d1e7c7"
MODEL_REV="0c0e3051f131929182e2c023b9537f8b1c68adfe"

class UnresolvedRealization(RuntimeError):pass

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def inspect_audio(path):
    y,sr=sf.read(path,dtype="float32",always_2d=True)
    if sr!=24000 or y.shape[1]!=1 or not len(y) or not np.isfinite(y).all():
        raise RuntimeError("malformed carrier waveform")
    mono=y[:,0]
    info={"sample_rate":sr,"frames":len(mono),"duration_s":len(mono)/sr,
          "peak":float(np.max(np.abs(mono))),"clipping_fraction":float(np.mean(np.abs(mono)>=.999)),
          "rms":float(np.sqrt(np.mean(mono**2))),"sha256":sha(path)}
    if info["clipping_fraction"]>1e-4 or info["rms"]<1e-4:
        raise RuntimeError("clipped or silent waveform")
    return mono,sr,info

def lock_runtime(engine,model,profile,anchor):
    engine=Path(engine).resolve();model=Path(model).resolve()
    profile=Path(profile).resolve();anchor=Path(anchor).resolve()
    for p in [engine,model/"model.safetensors",model/"speech_tokenizer/model.safetensors",profile,anchor]:
        if not p.is_file():raise FileNotFoundError(p)
    if sha(profile)!=PROFILE or sha(anchor)!=ANCHOR:raise RuntimeError("carrier hash mismatch")
    revision=subprocess.check_output(["git","-C",str(engine.parent),"rev-parse","HEAD"],text=True).strip()
    if revision!=ENGINE_REV:raise RuntimeError("engine revision mismatch")
    changes=subprocess.check_output(["git","-C",str(engine.parent),"diff","--name-only","HEAD"],text=True).strip()
    if changes:raise RuntimeError("unrecorded engine source modification")
    cfg=json.loads((model/"config.json").read_text())
    if cfg.get("tts_model_type")!="custom_voice":raise RuntimeError("wrong model type")
    # Fetch/build provenance must accompany the actual bytes; a directory name is insufficient.
    prov_path=model/"MARI_PINNED_MODEL_RECEIPT.json"
    if not prov_path.exists():raise RuntimeError("model byte provenance missing")
    prov=json.loads(prov_path.read_text())
    if prov.get("revision")!=MODEL_REV:raise RuntimeError("model revision provenance mismatch")
    for rel,h in prov["files"].items():
        if sha(model/rel)!=h:raise RuntimeError("model bytes changed: "+rel)
    return {"engine":str(engine),"engine_revision":revision,"engine_binary_sha256":sha(engine),
            "model":str(model),"model_revision":MODEL_REV,"model_hashes":prov["files"],
            "profile":str(profile),"profile_sha256":PROFILE,"anchor":str(anchor),"anchor_sha256":ANCHOR}

def carrier_render(text,output,lock,seed=88000,timeout=900):
    if not text.strip():raise ValueError("empty text")
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    cmd=[lock["engine"],"-d",lock["model"],"--load-voice",lock["profile"],"--xvector-only",
         "-l","English","--text",text,"--seed",str(seed),"--temperature","0.42",
         "--top-k","40","--top-p","0.95","--rep-penalty","1.05","-j4"]
    with tempfile.TemporaryDirectory(dir=output.parent) as td:
        tmp=Path(td)/"carrier.wav"
        cp=subprocess.run(cmd+["-o",str(tmp)],capture_output=True,text=True,timeout=timeout,
                          env={**os.environ,"OPENBLAS_NUM_THREADS":"4"})
        log=cp.stdout+cp.stderr
        if cp.returncode or not tmp.exists():raise RuntimeError("carrier failed; no fallback\n"+log[-2000:])
        _,_,info=inspect_audio(tmp)
        tmp.replace(output)
    return {"command":cmd+["-o",str(output)],"seed":seed,"audio":info,"log":log,
            "conditioning":"frozen_xvector_no_instructions","lock_hash":digest(lock)}

def realization_coverage(plan):
    """Each requested causal channel is accounted for, including unsupported ones."""
    kinds={e["event"]["kind"] for e in plan["journal"]}
    unresolved=[]
    for k in plan["trajectory"]:
        c=k["controls"]
        if c["precision"]>.56:unresolved.append("articulation_precision")
        if c["attack_softness"]>.30:unresolved.append("vocal_attack")
        if c["phonatory_tension"]>.20:unresolved.append("phonatory_tension")
        if k["thought"] in {"realizing","correcting","judging","withholding"}:
            unresolved.append("local_commitment_and_finality")
    if "nonlexical" in kinds:unresolved.append("situated_nonlexical_vocalization")
    if "interrupt" in kinds:unresolved.append("live_interruption_waveform_truncation")
    return {"realized":["continuous_carrier_identity","event_boundary_latency","pitch_preserving_timing","projection_gain"],
            "not_yet_realized":sorted(set(unresolved)),
            "not_established":["character_specific_person_presence","full_expressive_bandwidth","physical_vocal_coherence"],
            "prose_direction_to_renderer":False}

def temporal_render(carrier,output,plan,alignment,diagnostic=False):
    verify_plan(plan)
    coverage=realization_coverage(plan)
    if not diagnostic and (coverage["not_yet_realized"] or coverage["not_established"]):
        raise UnresolvedRealization("full performance admission denied: "+json.dumps(coverage))
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    y,sr,base=inspect_audio(carrier)
    aligned=alignment["words"]
    if len(aligned)!=len(plan["words"]):raise ValueError("word alignment is incomplete")
    if alignment.get("source_sha256")!=base["sha256"]:raise ValueError("alignment belongs to different audio")
    for a,w in zip(aligned,plan["words"]):
        if a["word"].lower().replace("’", "'")!=w.lower().replace("’", "'"):
            raise ValueError("alignment text mismatch")
        if not 0<=a["start"]<a["end"]<=base["duration_s"]+.05:raise ValueError("alignment time outside audio")
    if any(a["start"]>b["start"] for a,b in zip(aligned,aligned[1:])):raise ValueError("nonmonotonic alignment")
    # Native model realizes the complete utterance once. Only event-caused changes
    # affect timing; punctuation and words do not select affect.
    commands=[];last_rate=None
    knots=plan["trajectory"]
    for i,a in enumerate(aligned):
        rate=round(knots[i]["controls"]["rate"],4)
        if last_rate is None or abs(rate-last_rate)>=.002:
            commands.append(f"{a['start']:.5f} rubberband tempo {rate:.4f};")
            last_rate=rate
    # Gain is smoothly interpolated in the performed stream. No added breath/fry/noise.
    t=np.arange(len(y))/sr
    positions=[a["start"] for a in aligned]+[len(y)/sr]
    gains=[k["controls"]["gain_db"] for k in knots]
    envelope=np.interp(t,positions,gains)
    y=y*(10**(envelope/20)).astype(np.float32)
    with tempfile.TemporaryDirectory(dir=output.parent) as td:
        td=Path(td);inp=td/"gain.wav";out=td/"timed.wav";cmdfile=td/"tempo.txt"
        sf.write(inp,y,sr,subtype="FLOAT")
        cmdfile.write_text("\n".join(commands))
        cmd=["ffmpeg","-v","error","-y","-i",str(inp),"-af",
             "asendcmd=f="+str(cmdfile)+",rubberband=tempo=1:pitch=1:transients=crisp:detector=compound:phase=laminar:window=long",
             "-ar","24000","-ac","1","-c:a","pcm_s16le",str(out)]
        cp=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        if cp.returncode:raise RuntimeError("temporal realization failed: "+cp.stderr)
        z,_,_=inspect_audio(out)
        # Latency occurs at a causal beat onset, once per boundary. Alignment maps
        # its location through the tempo trajectory, never through an arbitrary pause grid.
        additions={}
        for e in plan["journal"]:
            event=e["event"];i=event["at_word"]
            if i>=len(aligned) or event["kind"] not in {"thought","mask","feedback"}:continue
            delay=knots[i]["controls"]["onset_delay_s"]
            if delay>.02:additions[i]=max(additions.get(i,0),delay)
        spans=[];start=0
        for i,delay in sorted(additions.items()):
            tm=0.0;prev=0.0
            for j in range(i+1):
                end=aligned[j]["start"]
                tm+=(end-prev)/knots[max(0,j-1)]["controls"]["rate"];prev=end
            index=min(len(z),int(tm*sr))
            # Use the preceding actual silence only; never cut into a voiced word.
            lo=max(start,index-int(.10*sr));hi=min(len(z),index+int(.02*sr))
            if hi>lo:
                energy=np.convolve(z[lo:hi]**2,np.ones(120)/120,mode="same")
                point=lo+int(np.argmin(energy))
                if energy[point-lo]<2e-5:
                    spans.extend([z[start:point],np.zeros(int(delay*sr),np.float32)]);start=point
                else:coverage.setdefault("unrealized_local_events",[]).append({"event":i,"reason":"no safe unvoiced insertion site"})
        spans.append(z[start:]);z=np.concatenate(spans)
        sf.write(out,z,sr,subtype="PCM_16")
        _,_,info=inspect_audio(out);out.replace(output)
    return {"schema":"mari-realization-receipt/1.0","status":"DIAGNOSTIC_RENDER",
            "plan_hash":plan["plan_hash"],"carrier":base,"output":info,"coverage":coverage,
            "tempo_commands":commands,"random_disfluency":False,"identity_changed":False,
            "performance_complete":False}
