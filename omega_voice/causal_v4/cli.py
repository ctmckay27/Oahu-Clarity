"""Inspectable compile/render interface with transactional continuity."""
import argparse
import json
import os
from pathlib import Path
import tempfile
from .runtime import compile_scene,verify_plan,digest
from .renderer import lock_runtime,carrier_render,temporal_render,realization_coverage,UnresolvedRealization
from .evaluate import Evaluator

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=".mari-")
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--text",required=True);ap.add_argument("--scene")
    ap.add_argument("--prior-state");ap.add_argument("--plan",required=True)
    ap.add_argument("--engine");ap.add_argument("--model");ap.add_argument("--profile");ap.add_argument("--anchor")
    ap.add_argument("--asr-model");ap.add_argument("--speaker-model");ap.add_argument("--output")
    ap.add_argument("--diagnostic",action="store_true");ap.add_argument("--seed",type=int,default=88000)
    a=ap.parse_args()
    scene=json.loads(Path(a.scene).read_text()) if a.scene else {}
    prior=json.loads(Path(a.prior_state).read_text()) if a.prior_state else None
    p=compile_scene(a.text,scene,prior);verify_plan(p);atomic_json(a.plan,p)
    if not a.output:
        print(json.dumps({"status":"COMPILED","plan_hash":p["plan_hash"],"coverage":realization_coverage(p)}));return
    if not a.diagnostic:
        # Check admission before any model cost or speculative output file.
        raise UnresolvedRealization("Current adapter has no completed-performance acceptance receipt; diagnostic execution is available.")
    required=[a.engine,a.model,a.profile,a.anchor,a.asr_model,a.speaker_model]
    if not all(required):raise ValueError("exact engine, model, carrier, and evaluator paths required")
    lock=lock_runtime(a.engine,a.model,a.profile,a.anchor)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as td:
        base=Path(td)/"base.wav";carrier=carrier_render(a.text,base,lock,a.seed)
        evaluator=Evaluator(a.asr_model,a.speaker_model,a.anchor)
        baseline=evaluator.evaluate(base,a.text)
        if not baseline["quality_screen_pass"] or not baseline["alignment"]["exact_words"]:
            atomic_json(str(output)+".failure.json",baseline)
            raise RuntimeError("baseline identity, intelligibility, or alignment failed")
        receipt=temporal_render(base,output,p,baseline["alignment"],diagnostic=True)
        receipt["evaluation"]=evaluator.evaluate(output,a.text)
        receipt["carrier_generation"]=carrier
    atomic_json(str(output)+".receipt.json",receipt)
    # A diagnostic must not silently become the continuing production state.
    atomic_json(str(output)+".diagnostic_state.json",p["final_state"])
    print(json.dumps({"status":"DIAGNOSTIC_ONLY","output":str(output),"receipt":str(output)+".receipt.json"}))

if __name__=="__main__":main()
