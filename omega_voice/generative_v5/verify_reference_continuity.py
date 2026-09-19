"""Discriminate accumulated reference drift from a text-specific carrier defect."""
import argparse,json,pathlib
from .native import render,anchor_reference
from .reference_context import anchored_history
from .session import durable_json
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha
from ..causal_v4.runtime import ANCHOR

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root)
 d=r/'continuation/reference_continuity';d.mkdir(exist_ok=True)
 parent=r/'continuation/causal_episodes/heldout_long_form/session/listener_concern'
 receipt=json.loads((parent/'receipt.json').read_text());audio=parent/'carrier.wav'
 if not receipt['quality_admitted'] or sha(audio)!=receipt['baseline']['audio']['sha256']:raise ValueError('parent not verified')
 prev={'audio':str(audio),'text':receipt['delivered_plan']['text'],'sha256':sha(audio),'source_carrier_sha256':ANCHOR,'role':'preceding verified native carrier'}
 text='The small silver key, please. That one opens the courtyard gate. Thank you. Stay here where it is warm; I should only be a few minutes.'
 combined=anchored_history(r,prev,d/'reference.wav')
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 rows=[]
 for name,ref in [('anchor_only',anchor_reference(r)),('anchored_history',combined)]:
  p=d/(name+'.wav');native=render(r,text,p,95103,reference=ref);q=ev.evaluate(p,text)
  rows.append({'id':name,'renderer':native,'quality':q,'full_completion':False});durable_json(d/'RESULTS.json',rows)
  print(name,q['wer'],q['speaker_similarity'],flush=True)
if __name__=='__main__':main()
