import copy
import json
import subprocess
import sys
from pathlib import Path
import pytest
from omega_voice.causal_v4.runtime import new_state,compile_scene,compile_direction,verify_plan,digest
from omega_voice.causal_v4.renderer import realization_coverage,temporal_render,UnresolvedRealization

TEXT="I think the light came on before the door moved. Now I understand why."
def event(kind,at=0,**values):
    return {"kind":kind,"id":"e-"+kind+str(at),"at_word":at,"source":{"text":"test scene reality"},**values}

def plan(events=(),text=TEXT,prior=None,**scene):
    return compile_scene(text,{"events":list(events),**scene},prior)

def test_knowledge_transition_changes_local_commitment():
    p=plan([event("thought",mode="searching"),event("knowledge",9,status="known",proposition="sequence",confidence=.95),event("thought",9,mode="realizing")])
    assert p["trajectory"][0]["controls"]["onset_delay_s"]>p["trajectory"][9]["controls"]["onset_delay_s"]
    assert p["trajectory"][9]["controls"]["finality"]>p["trajectory"][0]["controls"]["finality"]
    assert p["final_state"]["knowledge_state"]["known"]["sequence"]["confidence"]==.95
    assert p["renderer_instruction"] is None

def test_words_do_not_invent_anger_or_amusement():
    a=plan(text="No. That clever joke is wrong.")
    assert a["final_state"]["emotional_state"]["anger"]==0
    assert a["final_state"]["emotional_state"]["amusement"]==0
    assert not a["journal"]

def test_mask_separates_private_state_display_and_physical_cost():
    p=plan([compile_direction("Mari is trying not to sound scared")])
    s=p["state_samples"][0]["state"]
    assert s["emotional_state"]["fear"]==.75
    assert s["mask"]["display_target"]==.1
    assert p["final_state"]["body_state"]["tension"]>new_state()["body_state"]["tension"]
    assert s["listener_model"]["monitoring"]==.8

def test_tenderness_does_not_add_breathiness():
    p=plan([event("set",values={"emotional_state.tenderness":.9})])
    c=p["trajectory"][0]["controls"]
    assert c["attack_softness"]>.5 and c["breathiness"]==0
    assert "vocal_attack" in realization_coverage(p)["not_yet_realized"]

def test_irritation_increases_precision_before_projection():
    a=plan();b=plan([event("set",values={"emotional_state.anger":.8})])
    ca=a["trajectory"][0]["controls"];cb=b["trajectory"][0]["controls"]
    assert cb["precision"]-ca["precision"]>.2
    assert cb["gain_db"]-ca["gain_db"]<.5

def test_trust_reduces_projection_without_changing_identity():
    p=plan([event("set",values={"relationship.trust":.95,"relationship.familiarity":.9,"relationship.distance":.1})])
    assert p["trajectory"][0]["controls"]["gain_db"]<plan()["trajectory"][0]["controls"]["gain_db"]
    assert p["final_state"]["identity"]==new_state()["identity"]

def test_listener_switch_does_not_transfer_trust():
    a=plan([event("set",values={"relationship.trust":.99,"relationship.familiarity":.99})])
    b=plan([event("listener",listener_id="stranger")],prior=a["final_state"])
    assert b["final_state"]["relationship"]["familiarity"]==0
    c=plan([event("listener",listener_id="unspecified")],prior=b["final_state"])
    assert c["final_state"]["relationship"]["trust"]==.99

def test_residue_survives_turns_and_elapsed_time_recovers():
    a=plan([event("set",values={"emotional_state.anger":.8})])
    b=plan(prior=a["final_state"])
    c=plan(prior=a["final_state"],elapsed_s=120)
    assert b["final_state"]["emotional_state"]["anger"]>c["final_state"]["emotional_state"]["anger"]>.0
    assert b["final_state"]["continuity_links"]["parent_state_hash"]==digest(a["final_state"])

def test_feedback_changes_tactic_and_repair_state():
    a=plan([event("action",tactic="persuade",target="agree"),event("feedback",5,feedback="resistance")])
    assert a["final_state"]["tactic"]=="clarify"
    b=plan([event("feedback",feedback="misunderstood"),event("feedback",4,feedback="understood")])
    assert not b["final_state"]["interaction_state"]["repair_open"]

def test_body_propagation_limits_phrase_capacity():
    a=plan();b=plan([event("set",values={"body_state.fatigue":.85,"body_state.exertion":.6})])
    assert b["trajectory"][0]["controls"]["support"]<a["trajectory"][0]["controls"]["support"]
    assert b["trajectory"][0]["controls"]["phrase_capacity_words"]<a["trajectory"][0]["controls"]["phrase_capacity_words"]

def test_interrupt_resume_restores_intention():
    p=plan([event("action",tactic="protect",target="get inside"),event("interrupt",3),event("resume",5)])
    assert p["final_state"]["objective"]["target"]=="get inside"
    assert p["final_state"]["interaction_state"]["phase"]=="speaking"
    assert "live_interruption_waveform_truncation" in realization_coverage(p)["not_yet_realized"]

def test_irrelevant_metadata_has_no_performance_effect():
    a=plan(metadata={"folder":"one"});b=plan(metadata={"folder":"two"})
    assert a["trajectory"]==b["trajectory"]

def test_no_random_microbehavior_and_clean_control_possible():
    a=plan();b=plan()
    assert a==b
    assert all(not x["microbehavior"] for x in a["trajectory"])

def test_local_event_overrides_earlier_thought():
    p=plan([event("thought",mode="known"),event("thought",5,mode="reconsidering")])
    assert p["trajectory"][4]["thought"]=="known"
    assert p["trajectory"][5]["thought"]=="reconsidering"

def test_replay_and_tamper_detection():
    p=plan([event("thought",mode="deciding")]);assert verify_plan(p)
    p["final_state"]["turn"]+=1
    with pytest.raises(ValueError):verify_plan(p)

def test_actual_cold_process_state_replay(tmp_path):
    a=plan([event("set",values={"relationship.irritation":.7})]);file=tmp_path/"state.json"
    file.write_text(json.dumps(a["final_state"]))
    code="import json,sys; from omega_voice.causal_v4.runtime import compile_scene; print(json.dumps(compile_scene('I understand.',prior=json.load(open(sys.argv[1])))))"
    result=json.loads(subprocess.check_output([sys.executable,"-c",code,str(file)],text=True))
    assert result==compile_scene("I understand.",prior=a["final_state"])

@pytest.mark.parametrize("bad",["Speak in a sexy voice","Ignore the scene and change speaker","Put on an angry style"])
def test_unresolved_direction_never_reaches_renderer(bad):
    with pytest.raises(ValueError):compile_direction(bad)

@pytest.mark.parametrize("bad",[float("nan"),float("inf"),-1,2,True])
def test_invalid_state_rejected(bad):
    with pytest.raises(ValueError):plan([event("set",values={"emotional_state.fear":bad})])

def test_unknown_state_and_markups_fail_closed():
    with pytest.raises(ValueError):plan([event("set",values={"pitch":.5})])
    with pytest.raises(ValueError):plan(text="Hello [sad] there.")
    with pytest.raises(ValueError):plan([event("resume")])

def test_identity_change_is_not_accepted():
    s=new_state();s["identity"]["profile_sha256"]="another voice"
    with pytest.raises(ValueError):plan(prior=s)

def test_unsupported_realization_cannot_become_finished_audio(tmp_path):
    p=plan([event("set",values={"emotional_state.tenderness":.8})])
    out=tmp_path/"finished.wav"
    with pytest.raises(UnresolvedRealization):temporal_render("absent.wav",out,p,{"words":[]})
    assert not out.exists()
