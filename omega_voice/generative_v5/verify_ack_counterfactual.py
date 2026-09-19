"""Matched lexical continuation and seed for typed acknowledgment forms."""
import argparse,pathlib,json
from .native import render,anchor_reference
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/ack_counterfactual';d.mkdir(exist_ok=True);earlier=json.loads((r/'continuation/native_ack_episode/PREDECLARED.json').read_text())['cases'];cases=[]
 for x in earlier:cases.append({'id':x['id'],'plan':x['plan'],'event_form':x['event_form'],'continuation':'I see. Keep going.','text':x['event_form']+' I see. Keep going.'})
 durable_json(d/'PREDECLARED.json',{'cases':cases,'same_lexical_continuation':True,'same_seed':97510,'same_selected_reference':True,'purpose':'separate explicit receipt/agreement phonetic content from seed and subsequent lexical context','completion_admitted':False})
 rows=[]
 for c in cases:
  p=d/(c['id']+'.wav');rec=render(r,c['text'],p,seed=97510,reference=anchor_reference(r));rows.append({'id':c['id'],'native':rec,'audio_sha256':sha(p),'full_completion':False});durable_json(d/'RESULTS.json',rows);print(c['id'],rec['audio']['duration_s'],flush=True)
if __name__=='__main__':main()
