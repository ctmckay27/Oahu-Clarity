from omega_voice.causal_v4.runtime import compile_scene,verify_plan
def test_unavailable_memory_does_not_invent_a_search():
 e={'id':'name-unavailable','at_word':0,'kind':'knowledge','status':'unresolved','confidence':0.,'proposition':'the name','source':{'text':'Mari cannot remember the name.'}}
 p=compile_scene('I cannot give you the name.',{'events':[e]},policy='bounded_thought_recovery_v1')
 assert p['trajectory'][0]['thought']=='known'  # no search operation asserted
 # Missing a fact does not make a confident admission of ignorance tentative.
 # Assertion commitment needs its own source; do not paint uncertainty over
 # every statement merely because some world knowledge is unavailable.
 assert p['trajectory'][0]['controls']['finality']==.7
 assert p['final_state']['knowledge_state']['unresolved']==['the name']
 q=compile_scene('Her name is Alice.',{'events':[dict(e,id='name-recovered',status='known',confidence=.98,source={'text':'Mari finds the name in her notes.'})]},prior=p['final_state'],policy='bounded_thought_recovery_v1')
 assert q['final_state']['knowledge_state']['unresolved']==[]
 assert q['final_state']['knowledge_state']['known']['the name']['confidence']==.98
 assert verify_plan(p) and verify_plan(q)
