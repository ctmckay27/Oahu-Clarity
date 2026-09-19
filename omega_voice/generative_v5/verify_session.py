"""Audible diagnostic turns driven by actual, persisted conversational state."""
import argparse,json,pathlib
from .session import PerformanceSession
from .acoustics import measure
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--mode',choices=['native','physical'],default='native');ap.add_argument('--output',default='session_continuity');a=ap.parse_args();r=pathlib.Path(a.root)
 d=r/'continuation'/a.output;d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 bank=r/'continuation/isolated_calibration/isolated_finality_bank.npz'
 aligner=None
 if a.mode=='physical':
  from .alignment import ForcedAligner
  aligner=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json')
 def session(name):return PerformanceSession(r,d/name,ev,bank,sha(bank),mode=a.mode,aligner=aligner)
 events=[{'id':'location-unverified','at_word':0,'kind':'knowledge','status':'beliefs','confidence':.15,'proposition':'spare-key location','source':{'text':'Mari has only an unverified recollection of the location.'}},
         {'id':'memory-search','at_word':0,'kind':'thought','mode':'remembering','source':{'text':'Mari searches her memory while checking the room.'}},
         {'id':'key-seen','at_word':7,'kind':'knowledge','status':'known','confidence':.98,'proposition':'spare-key location','source':{'text':'At the second sentence Mari sees the key beside the blue bowl.'}},
         {'id':'realization','at_word':7,'kind':'thought','mode':'realizing','source':{'text':'Seeing the key resolves the search.'}}]
 rows=[]
 jobs=[('continuous','turn1','I thought the spare key was upstairs. I can see it beside the blue bowl.',{'events':events},94400),
       ('continuous','turn2','The key is beside the blue bowl. I will bring it to you.',{},94401),
       ('fresh_control','turn2','The key is beside the blue bowl. I will bring it to you.',{},94401)]
 for name,turn,text,scene,seed in jobs:
  receipt=d/name/turn/'receipt.json'
  try:
   if receipt.exists():result=json.loads(receipt.read_text())
   else:result=session(name).render_turn(turn,text,scene,seed,diagnostic=True)
   row={'session':name,'turn':turn,'receipt':str(receipt),'quality_admitted':result['quality_admitted'],
        'full_completion':False,'unresolved_channels':result['unresolved_channels']}
   if result['quality_admitted']:
    row.update(audio=result['trials'][-1]['audio'],acoustics=measure(result['trials'][-1]['audio']),
               state_time_s=result['delivered_plan']['final_state']['time_s'],
               parent_state_hash=result['parent_state_hash'])
  except Exception as e:row={'session':name,'turn':turn,'error':str(e),'full_completion':False}
  rows.append(row);(d/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
  if 'error' in row or not row['quality_admitted']:break
if __name__=='__main__':main()
