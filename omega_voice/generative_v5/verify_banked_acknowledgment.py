"""New text, changing knowledge, exact event-prefix independence and saved state."""
import argparse,pathlib,json,copy
from .session import PerformanceSession,durable_json
from .banked_acknowledgment import render_episode
from .native_event_bank import realize
from .alignment import ForcedAligner
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.runtime import new_state
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/banked_acknowledgment';d.mkdir(exist_ok=True);bank=r/'continuation/native_event_bank/BANK.json';ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');prop='The side door is open.';event={'id':'hear-door','behavior':'acknowledgment','intent':'receipt','proposition':prop,'source':{'kind':'authored_scene','listener_id':'Rowan','text':'Rowan reports the side door is open; Mari acknowledges receipt without independent assent.'}};durable_json(d/'PREDECLARED.json',{'source_bank_sha256':sha(bank),'cases':['receipt before independent checking','checked fact spoken','agreement after direct evidence'],'fresh_continuations':True,'no_Carl_tuning':True,'all_component_audio_preserved':True,'full_completion':False});results=[]
 def session():return PerformanceSession(r,d/'session',ev,mode='physical',aligner=al,temporal_policy='contrast_focus_v7',conditioning='selected_anchor',articulation=True,respiration=True)
 try:
  prior=new_state('event-invariance','Rowan');other=copy.deepcopy(prior);other['knowledge_state']['known']['The book is on the desk.']={'confidence':.9,'source':'independent irrelevant observation'};aa=realize(prior,event,bank,d/'irrelevant_a.wav');bb=realize(other,event,bank,d/'irrelevant_b.wav');irrelevant=(d/'irrelevant_a.wav').read_bytes()==(d/'irrelevant_b.wav').read_bytes()
  one=render_episode(session(),'heard',event,'I heard you. Let me check the door.',bank,seed=98400,diagnostic=True);results.append({'id':'heard','receipt':one});durable_json(d/'RESULTS.json',results);print('heard',one['quality']['speaker_similarity'],one['quality']['asr_text'],flush=True)
  scene={'events':[{'id':'checked-door','at_word':0,'kind':'knowledge','status':'known','proposition':prop,'confidence':.95,'source':{'kind':'authored_scene','text':'Mari directly checks the side door and sees it is open.'}}]};two=session().render_turn('checked','I checked the side door. It is open.',scene,seed=98401,diagnostic=True);results.append({'id':'checked','receipt':two});durable_json(d/'RESULTS.json',results);print('checked',two['quality_admitted'],flush=True)
  agreed=dict(event,id='agree-door',intent='agreement',source={'kind':'authored_scene','listener_id':'Rowan','text':'Rowan repeats the report; Mari has directly checked the door and agrees.'});three=render_episode(session(),'agreed',agreed,'That is what I found. We can use it.',bank,seed=98402,diagnostic=True);results.append({'id':'agreed','receipt':three});durable_json(d/'RESULTS.json',results);print('agreed',three['quality']['speaker_similarity'],three['quality']['asr_text'],flush=True)
  state=json.loads((d/'session/session_state.json').read_text())['state'];checks={'irrelevant_knowledge_event_audio_exact':irrelevant,'three_turns':state['turn']==3,'knowledge_retained':state['knowledge_state']['known'][prop]['confidence']==.95,'listener_retained':state['listener_model']['id']=='Rowan','heard_report_retained':prop in state['listener_model']['heard_propositions'],'event_components_exact':one['components_exact']and three['components_exact']};durable_json(d/'ASSESSMENT.json',{'checks':checks,'diagnostic_continuity_pass':all(checks.values()),'perceived_meaning_and_character_unverified':True,'full_completion':False})
 except Exception as e:durable_json(d/'ASSESSMENT.json',{'diagnostic_continuity_pass':False,'completed_turns':len(results),'error':repr(e),'full_completion':False});print('FAIL',repr(e),flush=True)
if __name__=='__main__':main()
