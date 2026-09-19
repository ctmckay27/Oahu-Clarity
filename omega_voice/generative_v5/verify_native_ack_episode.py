"""Cause-grounded acknowledgment phonology within the unchanged native speaker.

This successor tests the renderer's lexical phonetic path rather than empty text,
forced codes, or a stationary nasal source. Orthographic event forms are emitted
only from an explicit interaction event. They are never random hesitations.
The experiment does not admit perceived meaning or short-event identity.
"""
import argparse,pathlib,json,copy
from .native import render,anchor_reference
from .nonlexical import compile_event
from .session import durable_json
from ..causal_v4.runtime import new_state,digest
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/native_ack_episode';d.mkdir(exist_ok=True)
 proposition='The side door is open.';base=new_state('ack-native-qualification','Rowan')
 event={'id':'heard-side-door','behavior':'acknowledgment','intent':'receipt','proposition':proposition,'source':{'kind':'authored_scene','text':'Rowan has told Mari that the side door is open. She signals receipt without claiming independent agreement.'}}
 receipt=compile_event(base,event);known=copy.deepcopy(base);known['knowledge_state']['known'][proposition]={'confidence':.95,'source':'Mari directly checked the side door.'}
 agreement=compile_event(known,dict(event,id='checked-side-door',intent='agreement',source={'kind':'authored_scene','text':'Rowan reports that the side door is open; Mari already checked it and agrees.'}))
 cases=[{'id':'receipt','plan':receipt,'event_form':'Mm.','continuation':'Go on. I am following.'},{'id':'agreement','plan':agreement,'event_form':'Mm-hm.','continuation':'That matches what I saw.'}]
 for c in cases:c['text']=c['event_form']+' '+c['continuation']
 durable_json(d/'PREDECLARED.json',{'cases':cases,'mechanism':'explicit interaction event -> phonological nonlexical form plus ensuing lexical utterance -> native Talker with frozen Mari profile and anchor','competing_explanations':['native lexical path can realize brief acknowledgment phonology with speaker continuity','renderer spells letters or produces unrelated lexical speech','brief event is produced but its interaction meaning remains unqualified'],'gates':{'identity_min':.6013473320007324,'lexical_continuation_WER':0,'event_transcript_reported_without_normalization':True,'no_hidden_speaker':True},'limits':['No random disfluency or acting direction','No short-event identity claim from whole-utterance similarity','No agreement claim from receipt alone','No automatic production admission'],'implementation_sha256':sha(__file__),'full_completion':False})
 results=[]
 for i,c in enumerate(cases):
  out=d/(c['id']+'.wav');rf=out.with_suffix('.receipt.json')
  if out.exists():
   rec=json.loads(rf.read_text());assert rec['text']==c['text'] and rec['audio']['sha256']==sha(out)
  else:rec=render(r,c['text'],out,seed=97500+i,reference=anchor_reference(r))
  results.append({'id':c['id'],'plan_hash':digest(c['plan']),'native':rec,'audio_sha256':sha(out),'meaning_admitted':False,'short_event_identity_admitted':False,'full_completion':False});durable_json(d/'RESULTS.json',results);print(c['id'],rec['audio']['duration_s'],flush=True)
if __name__=='__main__':main()
