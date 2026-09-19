import json, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mari_performance import MariState, InteractionContext, plan_performance, controls_from_state

def test_irritation_tightens_before_breath_or_volume():
    z=MariState(irritation=.8,certainty=.6,restraint=.8)
    c=controls_from_state(z)
    assert c.articulation_precision > .45
    assert c.vocal_pressure > 0
    assert c.breathiness < 0

def test_intimacy_reduces_projection_not_turns_breathy():
    z=MariState(intimacy=.9,projection=0)
    c=controls_from_state(z)
    assert c.projection < 0
    assert c.breathiness < -.2

def test_search_to_resolution_is_temporal_not_global_label():
    p=plan_performance("Wait, maybe I missed something. Actually, I see it now. That's clever.")
    assert len(p.segments)>=3
    searches=[s.state.cognitive_search for s in p.segments]
    cert=[s.state.certainty for s in p.segments]
    assert max(searches)>searches[-1]
    assert cert[-1] > min(cert)

def test_amusement_enters_locally():
    p=plan_performance("The numbers are consistent. Actually, that's a very creative way to hide the error.")
    assert p.segments[-1].state.amusement > p.segments[0].state.amusement
    assert "amusement" in p.segments[-1].instruction.lower()

def test_context_carries_prior_state():
    prior=MariState(certainty=.8,restraint=.9,amusement=.4)
    p=plan_performance("Yes. Keep going.",context=InteractionContext(prior_state=prior,relationship="trusted"))
    assert p.initial_state.certainty==.8
    assert p.segments[0].state.intimacy>0

def test_negative_attractors_are_explicit():
    p=plan_performance("I know what happened. Tell me what changed after that.")
    for s in p.segments:
        x=s.instruction.lower()
        assert "generic sultry" in x and "anime-girl" in x and "assistant" in x

def test_plan_is_serializable():
    p=plan_performance("No. That's not the same problem, but I see why it looked similar.")
    json.dumps(p.to_dict())
