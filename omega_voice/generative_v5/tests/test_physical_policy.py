import numpy as np
from omega_voice.causal_v4.runtime import compile_scene
from omega_voice.generative_v5.physical import control_curves

def test_native_neutral_does_not_gain_controls_from_mismatched_runtime_policy():
 clock={'source_audio_sha256':'measured-fixture','duration_s':2.,'words':[{'start':.1,'end':.3},{'start':.9,'end':1.1}]}
 for policy in [None,'epistemic_focus_v2','embodied_continuity_v3','linguistic_scope_v4']:
  plan=compile_scene('Still here.',timeline=clock,policy=policy)
  assert all(np.array_equal(v,np.zeros_like(v)) for v in control_curves(plan,np.arange(401)*.005).values())

def test_matching_policy_preserves_real_relationship_effect():
 clock={'source_audio_sha256':'measured-fixture','duration_s':2.,'words':[{'start':.1,'end':.3},{'start':.9,'end':1.1}]}
 scene={'events':[{'kind':'set','id':'familiar','at_word':0,'values':{'relationship.trust':.95,'relationship.familiarity':.95,'relationship.distance':.1},'source':{'text':'Mari speaks with a trusted familiar listener.'}}]}
 plan=compile_scene('Still here.',scene,timeline=clock,policy='linguistic_scope_v4')
 curves=control_curves(plan,np.arange(401)*.005)
 assert curves['gain_db'].min()<-.5
 assert all(np.all(v[:20]==0) for v in curves.values())
