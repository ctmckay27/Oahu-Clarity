"""Event phonology without future lexical text in the renderer input."""
import argparse,pathlib,json
from .native import render,anchor_reference
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/standalone_ack';d.mkdir(exist_ok=True);cases=[{'id':'receipt','text':'Mm.'},{'id':'agreement','text':'Mm-hm.'}];durable_json(d/'PREDECLARED.json',{'cases':cases,'seed':98300,'frozen_anchor':True,'future_lexical_context_absent':True,'reason':'held-out native episode realized Hmm before checking-language despite requested receipt Mm; separate event production from later discovery text','no_standalone_biometric_pass_inferred':True,'full_completion':False});rows=[]
 for c in cases:
  p=d/(c['id']+'.wav');rec=render(r,c['text'],p,seed=98300,reference=anchor_reference(r));rows.append({'id':c['id'],'native':rec,'audio_sha256':sha(p),'full_completion':False});durable_json(d/'RESULTS.json',rows);print(c['id'],rec['audio']['duration_s'],flush=True)
if __name__=='__main__':main()
