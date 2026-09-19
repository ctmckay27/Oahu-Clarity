"""Delivered listener response and respiratory state across cold-opened turns."""
import argparse,pathlib,json
from .session import PerformanceSession,durable_json
from .alignment import ForcedAligner
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.runtime import digest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/listener_session';d.mkdir(exist_ok=True)
 def event(kind,source,**fields):return dict(kind=kind,id=kind+'-'+str(len(source)),at_word=0,source={'text':source},**fields)
 cases=[{'id':'concern','text':'You can leave the case here. I will check the landing before we move it.','scene':{'events':[
 event('listener','The current listener is Rowan.',listener_id='Rowan'),
 event('listener_condition','Rowan is worried about the unlit stairs.',values={'worry':.8}),
 event('set','Mari cares about Rowan and has little breath after carrying the case.',values={'relationship.concern':.8,'body_state.breath_reserve':.21,'body_state.exertion':.8,'body_state.fatigue':.6}),
 event('action','Mari chooses to reassure Rowan while checking the landing.',tactic='reassure',target='Rowan')]}},
 {'id':'clarification','text':'The smaller bag goes upstairs. The heavy case can stay beside the door.', 'scene':{'events':[
 event('listener_condition','Rowan is confused about which bag to carry.',values={'confusion':.8}),
 event('action','Mari clarifies which bag belongs upstairs.',tactic='clarify',target='Rowan')]}},
 {'id':'new_listener_scope_fix','text':'Please keep this door open. I need room to move the case.', 'scene':{'events':[
 event('listener','Mari turns to a porter she has just met.',listener_id='porter'),
 event('action','Mari requests practical help from the porter.',tactic='invite',target='porter'),
 event('speech_act','Mari asks the porter to keep the door open.',mode='request',until_word=5)]}}]
 durable_json(d/'PREDECLARED_CORPUS.json',cases)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');rows=[];prior=None
 for i,case in enumerate(cases):
  session=PerformanceSession(r,d/'session',ev,mode='physical',aligner=al,temporal_policy='listener_causal_v6',conditioning='selected_anchor',articulation=True,respiration=True)
  try:
   saved=d/'session'/case['id']/'receipt.json'
   if saved.exists():
    receipt=json.loads(saved.read_text())
    if not receipt.get('quality_admitted') or receipt['requested_scene']!=case['scene'] or receipt['delivered_plan']['text']!=case['text']:raise ValueError('existing receipt cannot be reused')
   else:receipt=session.render_turn(case['id'],case['text'],case['scene'],96700+i,diagnostic=True)
   p=receipt['delivered_plan'];q=receipt['trials'][-1]['evaluation'];continuity=prior is None or p['initial_state']==prior
   condition=p['final_state']['listener_model'].get('condition',{});expected=({'worry':.8},{'worry':.8,'confusion':.8},{})[i]
   checks={'quality':receipt['quality_admitted']and q['wer']==0,'cold_state_continuity':continuity,'person_specific_condition':condition==expected,'listener_worry_not_mari_fear':p['final_state']['emotional_state']['fear']==0}
   row={'id':case['id'],'checks':checks,'passed':all(checks.values()),'parent_state_hash':receipt['parent_state_hash'],'final_state_hash':digest(p['final_state']),'listener_condition':condition,'inhalations':len(receipt['respiration']['insertions']),'identity':q['speaker_similarity'],'wer':q['wer'],'alignment_error_s':receipt['trials'][-1]['alignment_error_s'],'full_completion':False};prior=p['final_state']
  except Exception as exc:row={'id':case['id'],'passed':False,'error':repr(exc),'full_completion':False}
  rows.append(row);durable_json(d/'RESULTS.json',rows);print(row,flush=True)
  if not row['passed']:break
if __name__=='__main__':main()
