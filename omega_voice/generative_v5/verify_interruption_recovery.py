"""Actual stream cancellation, heard-prefix commitment, and context recovery."""
import argparse,json,math,pathlib
import numpy as np,soundfile as sf
from .interaction import StreamingTurn
from .session import PerformanceSession,durable_json
from .alignment import ForcedAligner
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.runtime import compile_scene
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/interruption_recovery';d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json')
 text='I checked the front room and the kitchen. The package must be upstairs beside the wardrobe.'
 source={'text':'Mari is explaining where she has searched. The listener interrupts: I moved it to the car.'}
 def event(kind,at_word=0,**kw):return dict(id=kind+str(at_word),kind=kind,at_word=at_word,source=source,**kw)
 scene={'events':[event('assertion',proposition='package location',confidence=.4),
   event('action',tactic='share',target='find the package'),
   event('knowledge',8,proposition='unheard planned upstairs inference',status='beliefs',confidence=.7)]}
 plan=compile_scene(text,scene,policy='epistemic_focus_v2');durable_json(d/'intended_plan.json',plan)
 result={'full_completion':False,'scope':'actual interrupted delivery and same-speaker recovery; no person-presence judgment'}
 try:
  # Deterministic replay establishes a measurable gap for the external test
  # listener. The production cancellation mechanism receives no word schedule.
  pre=d/'preflight.wav';StreamingTurn(r,text,pre,lambda chunk,turn:None,94900).run()
  q=ev.evaluate(pre,text);result['preflight']=q
  if not q['quality_screen_pass'] or q['wer']!=0:raise ValueError('preflight intelligibility failed')
  aligned=al.align(ev.load(pre),text,sha(pre),True);result['preflight_alignment']=aligned
  low=aligned['words'][7]['end']+.02;high=aligned['words'][8]['start']-.02
  stop=math.ceil(low/.04)*.04
  if stop>high:raise ValueError('no verified delivery packet boundary between complete words')
  def listener(chunk,turn):
   if sum(map(len,turn.delivered))//2>=round(stop*24000):turn.interrupt(source)
  actual=d/'delivered.wav';stream=StreamingTurn(r,text,actual,listener,94900).run();result['stream']=stream
  aa,sr=sf.read(pre,dtype='int16');bb,sr2=sf.read(actual,dtype='int16')
  result['exact_prefix']=bool(sr==sr2 and np.array_equal(aa[:len(bb)],bb))
  if not result['exact_prefix']:raise ValueError('stream replay changed before interruption')
  session=PerformanceSession(r,d/'session',ev,mode='physical',aligner=al,temporal_policy='epistemic_focus_v2',conditioning='previous_carrier')
  interrupted=session.admit_interrupted_delivery('interrupted',plan,actual,actual.with_suffix('.receipt.json'),8,diagnostic=True)
  state=interrupted['delivered_plan']['final_state']
  result['unheard_fact_absent']='unheard planned upstairs inference' not in state['knowledge_state']['beliefs']
  result['intention_preserved']=state['interaction_state']['interrupted_intention']['target']=='find the package'
  recovery={'events':[event('resume'),event('knowledge',proposition='package location',status='known',confidence=.95),
    event('action',tactic='share',target='check the car') ]}
  resumed=session.render_turn('resumed','I see. Then I will check the car instead.',recovery,94901,diagnostic=True)
  result['recovery_quality_admitted']=resumed['quality_admitted'];result['recovery_state']=resumed['delivered_plan']['final_state']
  ref=json.loads((d/'session/resumed/carrier.receipt.json').read_text());result['reference_receipt']=ref
  tail,sr3=sf.read(d/'session/resumed/performance.wav',dtype='int16');assert sr3==sr
  joined=d/'episode.wav';sf.write(joined,np.concatenate([bb,tail]),sr,subtype='PCM_16')
  result['episode_quality']=ev.evaluate(joined,'I checked the front room and the kitchen. I see. Then I will check the car instead.')
  result['passed']=result['unheard_fact_absent'] and result['intention_preserved'] and result['recovery_quality_admitted'] and result['episode_quality']['quality_screen_pass'] and result['episode_quality']['wer']==0
 except Exception as e:
  result.update(passed=False,error=str(e))
  if hasattr(e,'report'):result['alignment_failure']=e.report
 finally:durable_json(d/'RESULTS.json',result)
 print(json.dumps({k:result.get(k) for k in ['passed','error','exact_prefix','unheard_fact_absent','intention_preserved','recovery_quality_admitted']},indent=2),flush=True)

if __name__=='__main__':main()
