import pytest
from omega_voice.causal_v4.runtime import compile_scene,verify_plan

def scene(mode,confidence,end=6):
 return {'events':[
  {'id':'belief','kind':'assertion','at_word':0,'proposition':'gate state','confidence':confidence,'source':{'text':'Mari has this explicitly specified confidence in the gate state.'}},
  {'id':'act','kind':'speech_act','at_word':0,'until_word':end,'mode':mode,'source':{'text':'Mari asks the listener to verify their action.'}}]}

def test_question_force_does_not_inherit_epistemic_finality():
 a=compile_scene('Did you leave the gate open?',scene('ask',.2),policy='linguistic_scope_v4')
 b=compile_scene('Did you leave the gate open?',scene('ask',.98),policy='linguistic_scope_v4')
 assert all(x['controls']==y['controls'] for x,y in zip(a['trajectory'][:-1],b['trajectory'][:-1]))
 assert a['final_state']['knowledge_state']['certainty']!=b['final_state']['knowledge_state']['certainty']
 assert 'scoped_act' not in a['final_state']['speech_behavior']
 assert verify_plan(a) and verify_plan(b)

def test_assertion_commitment_remains_effective_and_scope_expires():
 a=compile_scene('You left the gate open again.',scene('assert',.2),policy='linguistic_scope_v4')
 b=compile_scene('You left the gate open again.',scene('assert',.98),policy='linguistic_scope_v4')
 assert a['trajectory'][5]['controls']['finality']<b['trajectory'][5]['controls']['finality']
 p=compile_scene('Please check. It is open.',scene('request',.98,2),policy='linguistic_scope_v4')
 assert p['trajectory'][1]['controls']['finality']==.7
 assert p['trajectory'][4]['controls']['finality']==.98
 assert verify_plan(p)

def test_scope_requires_explicit_policy_and_valid_bounds():
 with pytest.raises(ValueError,match='requires linguistic_scope_v4'):
  compile_scene('Did you leave the gate open?',scene('ask',.2),policy='embodied_continuity_v3')
 with pytest.raises(ValueError,match='outside utterance'):
  compile_scene('Did you leave the gate open?',scene('ask',.2,9),policy='linguistic_scope_v4')
