from omega_voice.causal_v4.runtime import compile_scene,verify_plan

POLICY='bounded_thought_recovery_v1'
TEXT='I thought the key was upstairs. I can see it beside the blue bowl.'
EVENTS=[{'id':'search','at_word':0,'kind':'thought','mode':'remembering','source':{'text':'Mari searches her memory.'}},
 {'id':'seen','at_word':6,'kind':'knowledge','status':'known','confidence':.98,'proposition':'key location','source':{'text':'Mari sees the key.'}},
 {'id':'realization','at_word':6,'kind':'thought','mode':'realizing','source':{'text':'Seeing resolves the search.'}}]

def test_resolution_finishes_but_knowledge_remains():
 p=compile_scene(TEXT,{'events':EVENTS},policy=POLICY)
 assert p['trajectory'][6]['thought']=='realizing'
 assert p['final_state']['mental_state']['thought']=='known'
 assert p['final_state']['knowledge_state']['known']['key location']['confidence']==.98
 q=compile_scene('The key is here.',prior=p['final_state'],policy=POLICY)
 assert q['trajectory'][0]['thought']=='known'
 assert q['trajectory'][0]['controls']['finality']==.98
 assert verify_plan(p) and verify_plan(q)

def test_unresolved_search_persists_without_repeated_onsets():
 p=compile_scene(TEXT,{'events':EVENTS[:1]},policy=POLICY)
 assert p['final_state']['mental_state']['thought']=='remembering'
 assert p['trajectory'][0]['controls']['onset_delay_s']>0
 assert all(k['controls']['onset_delay_s']==0 for k in p['trajectory'][1:])

def test_completed_operation_recovers_during_listener_gap():
 p=compile_scene('Yes.',{'events':[dict(EVENTS[-1],at_word=0)]},policy=POLICY)
 q=compile_scene('I understand.',{'elapsed_s':2},prior=p['final_state'],policy=POLICY)
 assert q['trajectory'][0]['thought']=='known'

def test_legacy_plans_keep_their_exact_policy_and_replay():
 p=compile_scene(TEXT,{'events':EVENTS})
 assert 'temporal_policy' not in p
 assert p['final_state']['mental_state']['thought']=='realizing'
 assert verify_plan(p)
