from __future__ import annotations

import argparse, json, pathlib, subprocess
from typing import Optional

from authentic_performance import ActingContext, coach

def load_context(path: Optional[str]) -> ActingContext:
    if not path:
        return ActingContext()
    raw=json.loads(pathlib.Path(path).read_text())
    fields=set(ActingContext().__dict__)
    return ActingContext(**{k:v for k,v in raw.items() if k in fields})

def render(engine,model,profile,profile_mode,text,output,seed=120001,context=None,brief_output=None):
    for p in (engine,model,profile):
        if not pathlib.Path(p).exists():
            raise FileNotFoundError(str(p))
    brief=coach(text,context)
    if brief_output:
        pathlib.Path(brief_output).write_text(json.dumps(brief.to_dict(),indent=2))

    cmd=[str(engine),"-d",str(model),"--load-voice",str(profile)]
    if profile_mode=="xvector":
        cmd.append("--xvector-only")
    elif profile_mode=="graft":
        cmd.append("--icl-only")
    else:
        raise ValueError("profile_mode must be xvector or graft")
    cmd += [
        "-l","English",
        "--text",text,
        "--seed",str(seed),
        "--temperature","0.50",
        "--top-k","40",
        "--top-p","0.95",
        "--rep-penalty","1.05",
        "-j4",
        "-o",str(output),
        "--instruct",brief.renderer_instruction,
    ]
    cp=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
    if cp.returncode:
        raise RuntimeError("Mari authentic-performance renderer failed; no fallback used\n"+cp.stdout)
    return {
        "release":"Omega Voice / Mari Authentic Performance v3",
        "profile_mode":profile_mode,
        "single_continuous_renderer_call":True,
        "text":text,
        "playable_action":brief.playable_action,
        "output":output,
        "command":cmd,
    }

def main():
    ap=argparse.ArgumentParser(prog="omega-voice-v3")
    ap.add_argument("--engine",required=True)
    ap.add_argument("--model",required=True)
    ap.add_argument("--profile",required=True)
    ap.add_argument("--profile-mode",choices=["xvector","graft"],default="xvector")
    ap.add_argument("--text",required=True)
    ap.add_argument("--output")
    ap.add_argument("--seed",type=int,default=120001)
    ap.add_argument("--context-json")
    ap.add_argument("--brief-output")
    ap.add_argument("--coach-only",action="store_true")
    a=ap.parse_args()

    ctx=load_context(a.context_json)
    brief=coach(a.text,ctx)
    if a.coach_only:
        print(json.dumps(brief.to_dict(),indent=2))
        return
    if not a.output:
        raise SystemExit("--output required unless --coach-only")
    print(json.dumps(render(
        a.engine,a.model,a.profile,a.profile_mode,a.text,a.output,a.seed,ctx,a.brief_output
    )))

if __name__=="__main__":
    main()
