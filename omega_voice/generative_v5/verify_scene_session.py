"""Counterfactual natural-language scenes through the persisted diagnostic path."""
import argparse,json,pathlib
from .session import PerformanceSession
from .explicit_scene import ExplicitSceneCompiler
from .alignment import ForcedAligner
from .acoustics import measure
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 import spacy
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/scene_session';d.mkdir(exist_ok=True)
 compiler=ExplicitSceneCompiler(spacy.load('en_core_web_sm'))
 evaluator=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 aligner=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json')
 jobs=[('known','Mari already knows the answer.','I checked the receipt. It arrived on Tuesday.',94600),
       ('reconsider','Mari reconsiders.','I checked the receipt. It arrived on Tuesday.',94600),
       ('irrelevant','There is a green chair nearby. Mari already knows the answer.','I checked the receipt. It arrived on Tuesday.',94600),
       ('known_other_text','Mari knows the location.','The package is beside the window. I will bring it to you.',94601),
       ('reconsider_other_text','Mari reconsiders.','The package is beside the window. I will bring it to you.',94601)]
 rows=[]
 for name,scene,text,seed in jobs:
  session=PerformanceSession(r,d/name,evaluator,mode='physical',aligner=aligner,temporal_policy='bounded_thought_recovery_v1',scene_compiler=compiler)
  try:
   receipt=session.render_turn('turn1',text,scene,seed,diagnostic=True)
   audio=receipt['trials'][-1]['audio'];row={'id':name,'quality_admitted':receipt['quality_admitted'],'audio':audio,'sha256':sha(audio),'acoustics':measure(audio),'unresolved_channels':receipt['unresolved_channels'],'character_specificity':'not established','receipt':str(session.directory/'turn1/receipt.json')}
  except Exception as e:row={'id':name,'error':str(e),'quality_admitted':False}
  rows.append(row);(d/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
 checks={'all_quality_admitted':all(x['quality_admitted'] for x in rows),
   'irrelevant_scene_exact_waveform':rows[0].get('sha256')==rows[2].get('sha256') and 'sha256' in rows[0]}
 (d/'ASSESSMENT.json').write_text(json.dumps({'checks':checks,'full_completion':False,'scope':'bounded scene-to-waveform realization; not unrestricted scene or character qualification'},indent=2)+'\n')
if __name__=='__main__':main()
