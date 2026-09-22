from omega_voice.conversational_v6.runtime import new_conversation_state, plan_turn
from omega_voice.presence_v7.runtime import compile_presence, verify_presence

def _plan(**kw):
    s=new_conversation_state()
    return plan_turn(s,"You said it was upstairs.","Wait, no, I mean the blue one is downstairs.",intent={"thought":"correcting",**kw})

def test_all_seven_frontiers_present():
    p=_plan()
    v=compile_presence(p,semantic_focus=["blue","downstairs"],
        nonlexical=[{"kind":"mm","at_word":0,"cause":"acknowledgment before correction"}],blind_id="blind-01")
    assert verify_presence(v)
    assert v["thought_to_speech"]
    assert len(v["semantic_prominence"])==2
    assert v["microprosody"]
    assert "reserve_before" in v["embodied_continuity"]
    assert v["relationship_realization"]
    assert v["nonlexical_events"][0]["kind"]=="mm"
    assert v["blind_character_test"]["hide_condition_labels"] is True

def test_no_random_nonlexical():
    p=_plan()
    try:
        compile_presence(p,nonlexical=[{"kind":"hm"}])
        assert False
    except ValueError:
        pass

def test_identity_frozen():
    p=_plan()
    v=compile_presence(p)
    assert v["identity"]["change"] is False
    assert v["identity"]["anchor_sha256"]==p["state_observed"]["identity"]["anchor_sha256"]

def test_relationship_changes_projection_not_identity():
    s=new_conversation_state()
    p1=plan_turn(s,"Okay.","I know.",observation={"relationship":{"trust":.9,"familiarity":.9}})
    p2=plan_turn(s,"Okay.","I know.",observation={"relationship":{"trust":.2,"distance":.9}})
    a=compile_presence(p1); b=compile_presence(p2)
    assert a["relationship_realization"] != b["relationship_realization"]
    assert a["identity"] == b["identity"]

def test_searching_changes_thought_latency():
    s=new_conversation_state()
    a=plan_turn(s,"Where is it?","It is downstairs.",intent={"thought":"known"})
    b=plan_turn(s,"Where is it?","I think it is downstairs.",intent={"thought":"searching"})
    pa=compile_presence(a); pb=compile_presence(b)
    assert pb["thought_to_speech"][0]["latency_before_s"] > pa["thought_to_speech"][0]["latency_before_s"]
