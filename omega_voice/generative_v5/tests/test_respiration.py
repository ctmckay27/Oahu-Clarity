import pytest
from omega_voice.causal_v4.runtime import compile_scene,verify_plan,new_state
from omega_voice.generative_v5.respiration import phrase_boundaries

def test_inhalation_requires_time_and_conserves_volume_without_fatigue_reset():
 clock={'source_audio_sha256':'fixture','duration_s':1.,'words':[{'start':.4,'end':.8}]}
 prior=new_state();prior['body_state'].update(breath_reserve=.1,fatigue=.8)
 event={'id':'air','kind':'inhale','at_word':0,'duration_s':.3,'reserve_increment':.4,'source':{'text':'Physical reserve requires replenishment before speaking.'}}
 p=compile_scene('Ready.',{'events':[event]},prior,timeline=clock,policy='respiratory_budget_v5')
 assert p['final_state']['body_state']['breath_reserve']==pytest.approx(.49)
 assert p['final_state']['body_state']['fatigue']>.8
 assert verify_plan(p)
 with pytest.raises(ValueError,match='overlaps'):compile_scene('Ready.',{'events':[dict(event,duration_s=.5)]},prior,timeline=clock,policy='respiratory_budget_v5')
 with pytest.raises(ValueError,match='capacity'):compile_scene('Ready.',{'events':[dict(event,reserve_increment=.7)]},prior,timeline=clock,policy='respiratory_budget_v5')
 with pytest.raises(ValueError,match='actual following'):compile_scene('Ready.',{'events':[event]},prior,policy='respiratory_budget_v5')

def test_low_support_limits_pressure_without_changing_legacy_replay():
 prior=new_state();prior['body_state']['breath_reserve']=.05
 old=compile_scene('Ready.',prior=prior,policy='linguistic_scope_v4')
 new=compile_scene('Ready.',prior=prior,policy='respiratory_budget_v5')
 assert new['trajectory'][0]['controls']['gain_db']<old['trajectory'][0]['controls']['gain_db']
 assert new['final_state']['identity']==old['final_state']['identity']
 assert verify_plan(old) and verify_plan(new)
 assert phrase_boundaries('I checked the room, and the door. It is empty.')==[0,4,7,10]
