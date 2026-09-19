"""Held-out text and social combinations through cold-opened causal sessions."""
import argparse,pathlib,json
from .session import PerformanceSession,durable_json
from .alignment import ForcedAligner
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.runtime import digest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/contrast_session';d.mkdir(exist_ok=True)
 def event(kind,cause,**kw):return dict(id=kind+'-'+digest([cause,kw])[:12],kind=kind,at_word=0,source={'text':cause},**kw)
 first='The spare battery is in the green drawer. I checked it just now.';second='The meeting starts on Thursday. We still have time to finish this.'
 cases=[{'id':'clarify_location','text':first,'scene':{'events':[
 event('listener','Mari is speaking to Rowan.',listener_id='Rowan'),event('listener_condition','Rowan confused two drawers.',values={'confusion':.8}),event('set','Mari trusts Rowan and they work together.',values={'relationship.trust':.75,'relationship.familiarity':.6}),event('contrast','Rowan expected the battery in the blue drawer; Mari checked the green drawer.',expected='The spare battery is in the blue drawer.',observed='The spare battery is in the green drawer.',perspective='listener'),event('action','Mari specifies the drawer Rowan needs.',tactic='clarify',target='resolve the drawer misconception')]}},
 {'id':'correct_date','text':second,'scene':{'events':[
 event('feedback','Rowan understood the drawer correction.',feedback='understood'),event('listener_condition','Rowan is no longer confused about the drawers.',values={'confusion':0.}),event('set','Mari is somewhat tired and has responsibility for the schedule.',values={'body_state.fatigue':.2,'relationship.authority':.3}),event('contrast','Rowan expected the meeting on Tuesday; Mari has the schedule showing Thursday.',expected='The meeting starts on Tuesday.',observed='The meeting starts on Thursday.',perspective='listener'),event('action','Mari corrects the meeting date before planning the remaining work.',tactic='correct',target='correct the scheduling assumption')]}},
 {'id':'shared_amusement','text':first,'scene':{'events':[
 event('set','The familiar pair find their repeated drawer mix-up amusing.',values={'emotional_state.amusement':.8,'relationship.playfulness':.7,'relationship.trust':.8}),event('contrast','Both expected the blue drawer again, despite the battery having been in the green drawer.',expected='The spare battery is in the blue drawer.',observed='The spare battery is in the green drawer.',perspective='shared'),event('action','Mari gently teases Rowan about their shared repeated mistake.',tactic='tease',target='acknowledge the shared mismatch')]} }]
 durable_json(d/'PREDECLARED.json',{'cases':cases,'gates':{'WER':0,'identity_min':.6013473320007324,'word_clock_error_max_s':.08,'cold_state_continuity':True,'contrast_history_persists':True,'no_global_amusement_emphasis':True},'scope':'held-out causal prominence integration; no finished character review','full_completion':False})
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');rows=[];prior=None
 for i,case in enumerate(cases):
  session=PerformanceSession(r,d/'session',ev,mode='physical',aligner=al,temporal_policy='contrast_focus_v7',conditioning='selected_anchor',articulation=True,respiration=True,prominence=True,prominence_calibration=r/'continuation/matched_emphasis/SUMMARY.json')
  try:
   saved=d/'session'/case['id']/'receipt.json'
   if saved.exists():
    receipt=json.loads(saved.read_text())
    if not receipt.get('quality_admitted')or receipt['requested_scene']!=case['scene']:raise ValueError('existing case not admitted; investigate before resuming')
   else:receipt=session.render_turn(case['id'],case['text'],case['scene'],96900+i,diagnostic=True)
   p=receipt['delivered_plan'];q=receipt['trials'][-1]['evaluation'];history=[x for x in p['final_state']['character']['shared_history']if isinstance(x,dict)and x.get('kind')=='encountered_contrast'];controls=[x for x in p['trajectory']if x['controls']['emphasis']>0];checks={'quality':receipt['quality_admitted']and q['wer']==0,'cold_state_continuity':prior is None or prior==p['initial_state'],'one_lexical_focus':len(controls)==1,'contrast_history_persists':len(history)==i+1,'ephemeral_focus_cleared':'contrast_focus'not in p['final_state']['speech_behavior']}
   row={'id':case['id'],'checks':checks,'passed':all(checks.values()),'quality':q,'focus_words':[p['words'][x['at_word']]for x in controls],'prominence':receipt['prominence'],'parent_state_hash':receipt['parent_state_hash'],'final_state_hash':digest(p['final_state']),'unresolved_channels':receipt['unresolved_channels'],'full_completion':False};prior=p['final_state']
  except Exception as error:row={'id':case['id'],'passed':False,'error':repr(error),'full_completion':False}
  rows.append(row);durable_json(d/'RESULTS.json',rows);print(case['id'],row['passed'],row.get('error'),flush=True)
  if not row['passed']:break
if __name__=='__main__':main()
