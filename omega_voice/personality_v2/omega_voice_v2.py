from __future__ import annotations

import argparse, json, pathlib, subprocess, tempfile
from typing import Optional
import numpy as np
import soundfile as sf

from mari_performance import MariState, InteractionContext, plan_performance

def load_state(path: Optional[str]) -> Optional[MariState]:
    if not path: return None
    raw=json.loads(pathlib.Path(path).read_text())
    raw=raw.get("state",raw)
    keys=set(MariState().__dict__)
    return MariState(**{k:float(v) for k,v in raw.items() if k in keys})

def load_context(path: Optional[str], prior: Optional[MariState]) -> InteractionContext:
    if not path: return InteractionContext(prior_state=prior)
    raw=json.loads(pathlib.Path(path).read_text())
    return InteractionContext(
        relationship=str(raw.get("relationship","familiar")),
        publicness=float(raw.get("publicness",0.0)),
        urgency=float(raw.get("urgency",0.0)),
        stakes=float(raw.get("stakes",0.0)),
        conversational_goal=str(raw.get("conversational_goal","respond")),
        prior_state=prior,
        notes=list(raw.get("notes",[])),
    )

def build_cmd(engine,model,profile,profile_mode,text,instruction,wav,seed):
    cmd=[str(engine),"-d",str(model),"--load-voice",str(profile)]
    if profile_mode=="xvector":
        cmd.append("--xvector-only")
    elif profile_mode=="graft":
        cmd.append("--icl-only")
    else:
        raise ValueError("profile_mode must be xvector or graft")
    cmd += [
        "-l","English","--text",text,"--seed",str(seed),
        "--temperature","0.48","--top-k","40","--top-p","0.95",
        "--rep-penalty","1.05","-j4","-o",str(wav),
        "--instruct",instruction,
    ]
    return cmd

def render(engine,model,profile,profile_mode,text,output,seed=99000,state=None,context=None,plan_output=None,history_output=None):
    for p in (engine,model,profile):
        if not pathlib.Path(p).exists():
            raise FileNotFoundError(str(p))
    plan=plan_performance(text,state,context)
    if plan_output:
        pathlib.Path(plan_output).write_text(json.dumps(plan.to_dict(),indent=2))
    arrays=[]; sr=None; commands=[]
    with tempfile.TemporaryDirectory() as td:
        for seg in plan.segments:
            wav=pathlib.Path(td)/f"seg_{seg.index}.wav"
            cmd=build_cmd(engine,model,profile,profile_mode,seg.text,seg.instruction,wav,seed+seg.index)
            cp=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=480)
            if cp.returncode:
                raise RuntimeError("Mari Voice v2 renderer failed; no fallback used\n"+cp.stdout)
            y,r=sf.read(wav,dtype="float32",always_2d=True)
            y=y.mean(1); sr=sr or r
            if r!=sr: raise RuntimeError("sample-rate drift")
            arrays.append(y)
            if seg.index < len(plan.segments)-1:
                arrays.append(np.zeros(int(sr*seg.pause_after_ms/1000),dtype="float32"))
            commands.append(cmd)
    audio=np.concatenate(arrays)
    sf.write(output,audio,sr,subtype="PCM_16")
    if history_output:
        pathlib.Path(history_output).write_text(json.dumps({
            "state":plan.final_state.dict(),"last_text":text
        },indent=2))
    return {
        "release":"Omega Voice / Mari Voice personality layer v2",
        "profile_mode":profile_mode,
        "segments":len(plan.segments),
        "sample_rate":sr,
        "duration_s":len(audio)/sr,
        "output":output,
        "commands":commands,
    }

def main():
    ap=argparse.ArgumentParser(prog="omega-voice-v2")
    ap.add_argument("--engine",required=True)
    ap.add_argument("--model",required=True)
    ap.add_argument("--profile",required=True)
    ap.add_argument("--profile-mode",choices=["xvector","graft"],default="graft")
    ap.add_argument("--text",required=True)
    ap.add_argument("--output")
    ap.add_argument("--seed",type=int,default=99000)
    ap.add_argument("--state-json")
    ap.add_argument("--context-json")
    ap.add_argument("--history-json")
    ap.add_argument("--history-output")
    ap.add_argument("--plan-output")
    ap.add_argument("--plan-only",action="store_true")
    a=ap.parse_args()
    prior=load_state(a.history_json)
    state=load_state(a.state_json)
    ctx=load_context(a.context_json,prior)
    plan=plan_performance(a.text,state,ctx)
    if a.plan_only:
        print(json.dumps(plan.to_dict(),indent=2))
        return
    if not a.output:
        raise SystemExit("--output required unless --plan-only")
    print(json.dumps(render(
        a.engine,a.model,a.profile,a.profile_mode,a.text,a.output,a.seed,
        state,ctx,a.plan_output,a.history_output
    )))

if __name__=="__main__":
    main()
