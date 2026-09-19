import copy,pytest
from omega_voice.causal_v4.runtime import new_state
from omega_voice.generative_v5.nonlexical import compile_event

def event(intent='receipt'):
 return {'id':'listener-finished','behavior':'acknowledgment','intent':intent,'proposition':'gate open','source':{'text':'Listener has finished reporting the gate is open.'}}

def test_hearing_does_not_assert_truth_or_agreement():
 s=new_state();p=compile_event(s,event())
 assert p['final_state']['knowledge_state']==s['knowledge_state']
 assert p['final_state']['listener_model']['heard_propositions']==['gate open']
 assert p['final_state']['time_s']>s['time_s']
 assert p['final_state']['body_state']['breath_reserve']<s['body_state']['breath_reserve']
 with pytest.raises(ValueError,match='agreement requires'):compile_event(s,event('agreement'))

def test_unsupported_event_and_insufficient_breath_fail_closed():
 e=event();e['behavior']='laugh'
 with pytest.raises(ValueError,match='not implemented'):compile_event(new_state(),e)
 s=new_state();s['body_state']['breath_reserve']=.01
 with pytest.raises(ValueError,match='respiratory reserve'):compile_event(s,event())

def test_agreement_and_unrelated_fact_have_distinct_semantics():
 s=new_state();s['knowledge_state']['known']['gate open']={'confidence':.9,'source':{'text':'Direct observation.'}}
 a=compile_event(s,event('agreement'));q=copy.deepcopy(s);q['knowledge_state']['known']['unrelated']={'confidence':.99,'source':{'text':'Separate fact.'}}
 b=compile_event(q,event('agreement'))
 assert a['trajectory']==b['trajectory']
 assert a['final_state']['knowledge_state']!=b['final_state']['knowledge_state']
