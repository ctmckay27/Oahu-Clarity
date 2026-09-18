import argparse, json, os, pathlib, subprocess, tempfile, re
import numpy as np
import soundfile as sf

HERE=pathlib.Path(__file__).resolve().parent
CFG=json.loads((HERE/"omega_voice_v1.json").read_text())
PERFORMANCE={
 "neutral":"",
 "dry_amusement":"Dry understated amusement. A restrained intelligent smile in the timing, never cute, breathy, or theatrical.",
 "controlled_irritation":"Controlled irritation. Firmer consonants, contained force, clipped patience, no shouting and no theatrical anger.",
 "reassuring":"Calm direct reassurance. Grounded warmth, measured pace, steady clarity, never sentimental or whispery.",
 "focused_urgency":"Focused urgency. Purposeful faster pace, sharper emphasis, controlled intensity, no panic."
}
def split_text(text,limit=230):
    parts=[x.strip() for x in re.split(r'(?<=[.!?])\s+',text.strip()) if x.strip()]
    out=[];cur=""
    for s in parts:
        if cur and len(cur)+1+len(s)>limit: out.append(cur);cur=s
        else: cur=(cur+" "+s).strip()
    if cur: out.append(cur)
    return out or [text]
def resolve_runtime():
    engine=pathlib.Path(os.environ.get("OMEGA_VOICE_ENGINE", HERE/CFG["default_runtime"]["engine"]))
    model=pathlib.Path(os.environ.get("OMEGA_VOICE_MODEL", HERE/CFG["default_runtime"]["model"]))
    profile=HERE/CFG["profile"]
    for p in (engine,model,profile):
        if not p.exists(): raise FileNotFoundError(str(p))
    return engine,model,profile
def render(text,performance,output,seed=88000):
    if performance not in PERFORMANCE: raise ValueError(performance)
    engine,model,profile=resolve_runtime()
    arrays=[];sr=None;commands=[]
    with tempfile.TemporaryDirectory() as td:
        chunks=split_text(text)
        for i,ch in enumerate(chunks):
            wav=pathlib.Path(td)/f"{i}.wav"
            cmd=[str(engine),"-d",str(model),"--load-voice",str(profile),"--xvector-only","-l","English",
                 "--text",ch,"--seed",str(seed+i),"--temperature","0.42","--top-k","40","--top-p","0.95",
                 "--rep-penalty","1.05","-j4","-o",str(wav)]
            instr=PERFORMANCE[performance]
            if instr: cmd.extend(["--instruct",instr])
            cp=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=420)
            if cp.returncode: raise RuntimeError("Mari Voice v1 renderer failed; no fallback used\n"+cp.stdout)
            y,r=sf.read(wav,dtype="float32",always_2d=True);y=y.mean(1);sr=sr or r
            if r!=sr: raise RuntimeError("sample-rate drift")
            arrays.append(y)
            if i<len(chunks)-1: arrays.append(np.zeros(int(sr*.10),dtype="float32"))
            commands.append(cmd)
    audio=np.concatenate(arrays);sf.write(output,audio,sr,subtype="PCM_16")
    return {"release":CFG["release"],"performance":performance,"chunks":len(chunks),"sample_rate":sr,"duration_s":len(audio)/sr,"output":output}
def main():
    ap=argparse.ArgumentParser(prog="omega-voice")
    ap.add_argument("--text",required=True);ap.add_argument("--performance",choices=PERFORMANCE,default="neutral")
    ap.add_argument("--output",required=True);ap.add_argument("--seed",type=int,default=88000)
    a=ap.parse_args();print(json.dumps(render(a.text,a.performance,a.output,a.seed)))
if __name__=="__main__":main()