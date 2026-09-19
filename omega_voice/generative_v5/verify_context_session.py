"""Persisted thought-unit context with grounded proposition and prior state."""
import argparse,json,pathlib
from .session import PerformanceSession
from .explicit_scene import ExplicitSceneCompiler
from .alignment import ForcedAligner
from ..causal_v4.evaluate import Evaluator

def main():
 import spacy
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/context_session';d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');compiler=ExplicitSceneCompiler(spacy.load('en_core_web_sm'),policy='epistemic_focus_v2')
 def session(name):return PerformanceSession(r,d/name,ev,mode='physical',aligner=al,temporal_policy='epistemic_focus_v2',scene_compiler=compiler,conditioning='previous_carrier')
 context={'assertion':{'proposition':'package location','confidence':.3,'source':{'text':'The current question is where the package is. Mari has a provisional recollection.'}}}
 jobs=[('continuous','first','I thought the package was upstairs.',{'direction':'Mari is searching her memory.','context':context},94800),
 ('continuous','second','I can see it beside the window.',{'direction':'Mari sees the package and knows the answer.','context':context},94801),
 ('continuous','third','I will bring it to you. Keep the door open.',{},94802)]
 rows=[]
 for name,unit,text,scene,seed in jobs:
  try:
   receipt=session(name).render_turn(unit,text,scene,seed,diagnostic=True)
   row={'unit':unit,'quality_admitted':receipt['quality_admitted'],'final_state':receipt['delivered_plan']['final_state'],'parent_state_hash':receipt['parent_state_hash'],'unresolved_channels':receipt['unresolved_channels'],'audio':receipt['trials'][-1]['audio'],'receipt':str(d/name/unit/'receipt.json')}
  except Exception as e:row={'unit':unit,'quality_admitted':False,'error':str(e)}
  rows.append(row);(d/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n');print(unit,row['quality_admitted'],row.get('error'),row.get('unresolved_channels'),flush=True)
  if not row['quality_admitted']:break
 (d/'ASSESSMENT.json').write_text(json.dumps({'all_units_passed':len(rows)==len(jobs) and all(x['quality_admitted'] for x in rows),'full_completion':False,'scope':'native prior-carrier conditioning plus source-bound evolving physical state; no duplicate physical processing'},indent=2)+'\n')
if __name__=='__main__':main()
