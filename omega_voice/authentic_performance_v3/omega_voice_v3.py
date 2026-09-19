from __future__ import annotations

import argparse, json, pathlib, subprocess, sys, tempfile
from typing import Optional

from authentic_performance import ActingContext, coach
from scene_contract import Finding, SceneValidationError

def load_context(path: Optional[str]) -> ActingContext:
    if not path:
        return ActingContext()
    try:
        raw=json.loads(pathlib.Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise SceneValidationError([Finding("context", "invalid_json", str(exc), path)]) from exc
    return ActingContext.from_dict(raw)

def render(engine,model,profile,profile_mode,text,output,seed=120001,context=None,brief_output=None):
    brief=coach(text,context)
    for p in (engine,model,profile):
        if not pathlib.Path(p).exists():
            raise FileNotFoundError(str(p))
    inputs={pathlib.Path(p).resolve() for p in (engine,model,profile)}
    outputs=[pathlib.Path(p).resolve() for p in (output,brief_output) if p]
    if any(p in inputs for p in outputs) or len(set(outputs)) != len(outputs):
        raise ValueError("Output paths must be distinct from renderer inputs and each other")
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
    # Stage output so a failed renderer cannot leave an old WAV looking like a
    # new success or overwrite the caller's last successful artifact.
    target=pathlib.Path(output)
    with tempfile.TemporaryDirectory(prefix="mari-v3-", dir=target.parent) as work:
        staged=pathlib.Path(work)/target.name
        cmd[cmd.index("-o")+1]=str(staged)
        cp=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
        if cp.returncode:
            raise RuntimeError("Mari authentic-performance renderer failed; no fallback used\n"+cp.stdout)
        if not staged.is_file() or staged.stat().st_size == 0:
            raise RuntimeError("Mari renderer returned no output; no fallback used")
        staged.replace(target)
    return {
        "release":"Omega Voice / Mari Authentic Performance v3",
        "profile_mode":profile_mode,
        "single_continuous_renderer_call":True,
        "text":text,
        "playable_action":brief.playable_action,
        "output":output,
        "command":cmd,
        "validation":brief.validation,
        "evidence":{
            "renderer_invocations":1,
            "renderer_process":"completed",
            "output_file":"produced",
            "acoustic_performance":"not_evaluated",
            "permanent_voice_acceptance":"not_evaluated",
        },
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

    try:
        ctx=load_context(a.context_json)
        if a.coach_only:
            print(json.dumps(coach(a.text,ctx).to_dict(),indent=2))
            return
        if not a.output:
            raise SystemExit("--output required unless --coach-only")
        print(json.dumps(render(
            a.engine,a.model,a.profile,a.profile_mode,a.text,a.output,a.seed,ctx,a.brief_output
        )))
    except SceneValidationError as exc:
        print(json.dumps(exc.to_dict(),indent=2),file=sys.stderr)
        raise SystemExit(2) from exc

if __name__=="__main__":
    main()
