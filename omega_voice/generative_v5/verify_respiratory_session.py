"""Cold-reconstructed two-turn physical delivery with explicit breathing state."""
import argparse,pathlib,json
from .session import PerformanceSession,durable_json
from .alignment import ForcedAligner
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.runtime import digest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/respiratory_session';d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json')
 cases=[{'id':'carrying','text':'Leave the heavy case beside the door. I can carry the smaller bag up the stairs.',
  'scene':{'events':[{'kind':'set','id':'physical-demand','at_word':0,'values':{'body_state.breath_reserve':.21,'body_state.exertion':.8,'body_state.fatigue':.6},
   'source':{'text':'Mari has just carried the heavy case upstairs and has little respiratory reserve remaining.'}}]}},
 {'id':'checking','text':'Wait here while I check the landing. The light may be out, and I want to see the steps.',
  'scene':{'events':[{'kind':'action','id':'inspect-steps','at_word':0,'tactic':'clarify','target':'keep the listener stationary while Mari checks the stairs',
   'source':{'text':'The listener asks whether to follow immediately. Mari asks them to wait while she checks the landing.'}}]}}]
 durable_json(d/'PREDECLARED_CORPUS.json',cases);rows=[];previous=None
 for i,case in enumerate(cases):
  # New object per turn; no hidden in-memory state can supply continuity.
  session=PerformanceSession(r,d/'session',ev,mode='physical',aligner=al,temporal_policy='respiratory_budget_v5',conditioning='selected_anchor',articulation=True,respiration=True)
  try:
   receipt=session.render_turn(case['id'],case['text'],case['scene'],95600+i,diagnostic=True)
   plan=receipt['delivered_plan'];continuity=previous is None or plan['initial_state']==previous
   if not continuity:raise ValueError('respiratory state reset at cold reconstruction')
   previous=plan['final_state'];q=receipt['trials'][-1]['evaluation']
   row={'id':case['id'],'passed':receipt['quality_admitted'] and continuity,'parent_state_hash':receipt['parent_state_hash'],
    'final_state_hash':digest(previous),'inhalations':len(receipt['respiration']['insertions']),'initial_reserve':plan['initial_state']['body_state']['breath_reserve'],
    'final_reserve':previous['body_state']['breath_reserve'],'identity':q['speaker_similarity'],'wer':q['wer'],'alignment_error_s':receipt['trials'][-1]['alignment_error_s'],'full_completion':False}
  except Exception as exc:row={'id':case['id'],'passed':False,'error':str(exc),'full_completion':False}
  rows.append(row);durable_json(d/'RESULTS.json',rows);print(row,flush=True)
  if not row['passed']:break
if __name__=='__main__':main()
