import pytest
from omega_voice.causal_v4.runtime import compile_scene,verify_plan
POLICY='epistemic_focus_v2'
TEXT='The package is beside the window.'
def event(kind,**kw):return dict(id=kind,kind=kind,at_word=0,source={'text':'Explicit fixture scene evidence.'},**kw)
def plan(events,text=TEXT,prior=None):return compile_scene(text,{'events':events},prior,policy=POLICY)
def controls(p):return [k['controls'] for k in p['trajectory']]
def test_unrelated_knowledge_has_no_arbitrary_vocal_effect():
 p=plan([]);q=plan([event('knowledge',proposition='capital of France',confidence=.99,status='known')])
 assert controls(p)==controls(q)
 assert q['final_state']['knowledge_state']['known']['capital of France']['confidence']==.99
 assert verify_plan(q)
def test_relevant_evidence_updates_active_assertion_and_persists():
 focus=event('assertion',proposition='package location',confidence=.2)
 p=plan([focus]);q=plan([focus,event('knowledge',proposition='package location',confidence=.98,status='known')])
 assert p['trajectory'][0]['controls']['finality']==.2
 assert q['trajectory'][0]['controls']['finality']==.98
 later=plan([],prior=q['final_state']);assert later['trajectory'][0]['controls']['finality']==.98
 assert verify_plan(later)
def test_realization_does_not_invent_certainty():
 focus=event('assertion',proposition='package location',confidence=.2)
 q=plan([focus,event('thought',mode='realizing')])
 assert q['final_state']['knowledge_state']['certainty']==.2
def test_admitting_ignorance_can_be_confident():
 q=plan([event('knowledge',proposition='package location',confidence=0,status='unresolved'),
         event('assertion',proposition='package location',mode='admit_unknown',confidence=.98)],'I do not know where it is.')
 assert q['trajectory'][0]['controls']['finality']==.98
 assert q['final_state']['knowledge_state']['unresolved']==['package location']
def test_focus_requires_evidence_and_new_policy_is_explicit():
 focus=event('assertion',proposition='package location')
 with pytest.raises(ValueError,match='needs proposition evidence'):plan([focus])
 with pytest.raises(ValueError,match='requires epistemic_focus_v2'):compile_scene(TEXT,{'events':[focus]})
 p=compile_scene(TEXT,{'events':[event('knowledge',proposition='unrelated',confidence=.99,status='known')]},policy='bounded_thought_recovery_v1')
 assert p['trajectory'][0]['controls']['finality']==.99 and verify_plan(p)

def test_interrupted_prefix_preserves_policy_and_excludes_future_fact():
 from omega_voice.generative_v5.interaction import commit_interrupted_prefix
 focus=event('assertion',proposition='package location',confidence=.2)
 future=dict(event('knowledge',proposition='package location',confidence=.98,status='known'),at_word=4)
 p=plan([focus,future]);clock={'source_audio_sha256':'observed-prefix','duration_s':.8,'words':[{'start':0.,'end':.2},{'start':.2,'end':.7}]}
 q=commit_interrupted_prefix(p,2,clock,{'text':'Listener interrupts after the second word.'})
 assert q['temporal_policy']==POLICY and verify_plan(q)
 assert q['final_state']['knowledge_state']['certainty']==.2
 assert 'package location' not in q['final_state']['knowledge_state']['known']
