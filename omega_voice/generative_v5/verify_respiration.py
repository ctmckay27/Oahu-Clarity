"""Matched-waveform respiratory conservation and delivery checks."""
import argparse,pathlib,json,copy
from .respiration import realize
from .session import durable_json,timeline_from_evaluation
from .alignment import ForcedAligner
from ..causal_v4.runtime import new_state
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/respiratory_budget';d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');rows=[]
 for name in ['initial','listener_concern']:
  sd=r/f'continuation/causal_episodes/heldout_long_form/session/{name}';parent=json.loads((sd/'receipt.json').read_text());carrier=sd/'carrier.wav'
  if sha(carrier)!=parent['baseline']['audio']['sha256']:raise ValueError('carrier provenance changed')
  text=parent['delivered_plan']['text'];clock=timeline_from_evaluation(dict(parent['baseline'],alignment=parent['alignment']))
  neutral=new_state();depleted=new_state();depleted['body_state'].update(breath_reserve=.22,exertion=.8,fatigue=.6)
  for label,prior,scene in [('normal',neutral,{}),('depleted',depleted,{}),('irrelevant',depleted,{'metadata':{'unrelated_object_color':'blue'}})]:
   out=d/f'{name}_{label}.wav'
   try:
    receipt=realize(carrier,out,text,scene,prior,clock);q=ev.evaluate(out,text)
    aligned=al.align(ev.load(out),text,sha(out),q['wer']==0)
    error=max(abs(actual[k]-pred[k]) for actual,pred in zip(aligned['words'],receipt['delivered_timeline']['words']) for k in ['start','end'])
    row={'source_id':name,'condition':label,'quality':q,'alignment_error_s':error,'quality_pass':q['quality_screen_pass'] and q['wer']==0 and error<=.08,
     'inhalations':len(receipt['insertions']),'added_samples':receipt['added_samples'],'minimum_reserve':min(x['state']['body_state']['breath_reserve'] for x in receipt['plan']['state_samples']),
     'audio_sha256':sha(out),'receipt_sha256':sha(out.with_suffix('.respiration.json')),'full_completion':False}
    if label=='normal':row['exact_no_event_parity']=sha(out)==sha(carrier)
    if label=='irrelevant':row['irrelevant_exact_parity']=sha(out)==sha(d/f'{name}_depleted.wav')
   except Exception as exc:row={'source_id':name,'condition':label,'quality_pass':False,'error':str(exc),'full_completion':False}
   rows.append(row);durable_json(d/'RESULTS.json',rows);print(name,label,row.get('inhalations'),row['quality_pass'],row.get('error'),flush=True)
if __name__=='__main__':main()
