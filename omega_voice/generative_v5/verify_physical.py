"""Matched-waveform tests of causal physical realization and irrelevant controls."""
import argparse,json,pathlib
import numpy as np
from .physical import realize
from .session import timeline_from_evaluation
from .attack_calibration import attack_measure
from .acoustics import measure
from ..causal_v4.runtime import compile_scene
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/physical_heldout';d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 sources=[x for x in json.loads((r/'continuation/reference_heldout/RESULTS.json').read_text()) if x['condition']=='base']
 conditions=[('neutral',{}),('tenderness',{'emotional_state.tenderness':.8}),
             ('contained_irritation',{'relationship.irritation':.8,'emotional_state.anger':.4}),
             ('familiar_trust',{'relationship.trust':.95,'relationship.familiarity':.95,'relationship.distance':.1})]
 rows=[]
 for i,source in enumerate(sources):
  source_path=pathlib.Path(source['audio']);base=source['result'];timeline=timeline_from_evaluation(base)
  for name,values in conditions:
   scene={'events':[{'id':'physical-state','at_word':0,'kind':'set','values':values,'source':{'text':{'tenderness':'Mari feels tenderness toward the listener.','contained_irritation':'Mari is irritated but chooses contained force and precision.','familiar_trust':'Mari trusts this familiar listener and speaks at close conversational distance.'}.get(name,'No scene change.')}}]} if values else {}
   plan=compile_scene(source['text'],scene,timeline=timeline);out=d/f'{i}_{name}.wav'
   if not out.exists():realize(source_path,out,plan)
   result=ev.evaluate(out,source['text']);result.update(attack=attack_measure(out),acoustics=measure(out))
   rows.append({'text':source['text'],'condition':name,'audio':str(out),'plan':plan,'result':result})
   (d/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n')
   print(i,name,'speaker',result['speaker_similarity'],'WER',result['wer'],'attack',result['attack']['initial_to_body_db'],flush=True)
  irrelevant=compile_scene(source['text'],{'metadata':{'chair_color':'blue'}},timeline=timeline)
  out=d/f'{i}_irrelevant.wav'
  if not out.exists():realize(source_path,out,irrelevant)
  if sha(out)!=sha(source_path):raise RuntimeError('irrelevant fact changed physical waveform')
if __name__=='__main__':main()
