from omega_voice.causal_v4.runtime import compile_scene,verify_plan

def test_ctc_blanks_do_not_invent_breath_or_physical_recovery():
 clock={'source_audio_sha256':'measured-fixture','duration_s':2.,'words':[{'start':.1,'end':.3},{'start':.9,'end':1.1}]}
 old=compile_scene('Still here.',timeline=clock,policy='epistemic_focus_v2')
 new=compile_scene('Still here.',timeline=clock,policy='embodied_continuity_v3')
 assert old['final_state']['body_state']['breath_reserve']>old['initial_state']['body_state']['breath_reserve']
 assert new['final_state']['body_state']['breath_reserve']<new['initial_state']['body_state']['breath_reserve']
 assert new['final_state']['body_state']['fatigue']>new['initial_state']['body_state']['fatigue']
 assert verify_plan(old) and verify_plan(new)

def test_actual_between_turn_rest_remains_separate_from_alignment_gaps():
 a=compile_scene('I checked the room.',policy='embodied_continuity_v3')
 b=compile_scene('It is empty.',{'elapsed_s':2.},a['final_state'],policy='embodied_continuity_v3')
 c=compile_scene('It is empty.',{},a['final_state'],policy='embodied_continuity_v3')
 assert b['final_state']['body_state']['breath_reserve']>c['final_state']['body_state']['breath_reserve']
 assert verify_plan(b) and verify_plan(c)
