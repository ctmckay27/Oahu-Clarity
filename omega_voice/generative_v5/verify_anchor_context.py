"""Discriminate insufficient short-prefix conditioning from identity replacement."""
import argparse,pathlib
from .native import render,anchor_reference
from .session import durable_json
from ..causal_v4.evaluate import Evaluator

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/anchor_context';d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');rows=[]
 cases=[('failed_prefix','I thought the package was upstairs,',95000),('heldout_short','That is exactly what I meant.',95201),('heldout_question','Did you leave the gate open?',95202)]
 for name,text,seed in cases:
  for condition in ['profile','anchor_icl']:
   if name=='failed_prefix' and condition=='profile':audio=r/'continuation/causal_episodes/mid_sentence/session/recollection/carrier.wav'
   else:
    audio=d/(name+'_'+condition+'.wav');render(r,text,audio,seed,reference=anchor_reference(r) if condition=='anchor_icl' else None)
   q=ev.evaluate(audio,text);rows.append({'id':name,'condition':condition,'text':text,'seed':seed,'audio':str(audio),'quality':q,'full_completion':False});durable_json(d/'RESULTS.json',rows)
   print(name,condition,q['wer'],q['speaker_similarity'],flush=True)
 durable_json(d/'ASSESSMENT.json',{'all_anchor_icl_quality':all(x['quality']['quality_screen_pass'] and x['quality']['wer']==0 for x in rows if x['condition']=='anchor_icl'),'full_completion':False,'scope':'selected-anchor conditioning on short cold-start utterances; no identity selection or perceptual-presence claim'})
if __name__=='__main__':main()
