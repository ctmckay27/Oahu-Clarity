from omega_voice.conversational_v6 import new_conversation_state,plan_turn
from omega_voice.presence_v7 import compile_presence
from omega_voice.conversational_realization_v1 import build_realization_plan
from omega_voice.acoustic_actuator_v2.runtime import compile_actuation,envelope_sweep,blind_manifest

def sample():
 s=new_conversation_state()
 cp=plan_turn(s,"Which one?","I meant the blue one, actually.",intent={"thought":"correcting"})
 pp=compile_presence(cp,semantic_focus=["blue"],nonlexical=[{"kind":"mm","at_word":0,"cause":"acknowledgment"}])
 rp=build_realization_plan(cp)
 return cp,pp,rp

def test_couples_same_source_and_identity():
 _,p,r=sample(); a=compile_actuation(r,p)
 assert a["identity"]["change"] is False
 assert a["laws"]["speaker_profile_constant"] is True

def test_focus_reaches_actuator():
 _,p,r=sample(); a=compile_actuation(r,p)
 assert any(u["expanded_controls"]["prominence_strength"]>0 for u in a["units"])

def test_bounds():
 _,p,r=sample(); a=compile_actuation(r,p)
 for u in a["units"]:
  assert .90<=u["controls"]["rate"]<=1.07
  assert .92<=u["controls"]["volume"]<=1.06
  assert .46<=u["expanded_controls"]["onset_energy"]<=.58

def test_envelope_and_blind_manifest():
 _,p,r=sample(); a=compile_actuation(r,p)
 s=envelope_sweep(a); b=blind_manifest(a)
 assert len(s)==25
 assert len(b["conditions"])==25
 assert b["reveal_only_after_judgment"] is True
