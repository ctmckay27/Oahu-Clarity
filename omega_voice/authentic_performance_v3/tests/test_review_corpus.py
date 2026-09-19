"""Requirements declared in review/*.json before measuring admission.

A blocked harmless case is a coverage limitation, not a semantic true negative.
The executable fixture tests admission and transport, not speech performance.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from test_renderer import engine_fixture  # shared executable protocol fixture
from authentic_performance import ActingContext, coach
from omega_voice_v3 import render
from scene_contract import ActionBeat, SceneValidationError

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("review_evaluation", ROOT / "review/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
CASES = [case for name in ("corpus.json", "composition_probes.json")
         for case in json.loads((ROOT / "review" / name).read_text())["cases"]]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_predeclared_contract_cases(case):
    result = evaluation.probe(case)
    assert evaluation.meets_expectation(case, result), (case["reason"], result)
    if result["outcome"] == "accepted":
        assert result["compilation"]["dialogue_unchanged"]
        if case["field"] != "text":
            assert result["compilation"]["original_in_instruction"]
    else:
        assert result["findings"]
        assert all(f["value"] == case["value"] for f in result["findings"])
        if case["class"] in {"valid_unsupported", "unresolved"}:
            assert all(f["status"] == "unresolved" for f in result["findings"])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_corpus_renderer_admission_and_transport(case, engine_fixture, tmp_path):
    text, context = evaluation.case_input(case)
    engine_fixture["text"] = text
    output = Path(engine_fixture["output"])
    output.write_bytes(b"previous-output")
    if case["expect"] != "accepted":
        with pytest.raises(SceneValidationError):
            render(**engine_fixture, context=context)
        assert not (tmp_path / "calls.jsonl").exists()
        assert output.read_bytes() == b"previous-output"
        return
    expected = coach(text, context)
    render(**engine_fixture, context=context)
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0][calls[0].index("--text") + 1] == text
    assert calls[0][calls[0].index("--instruct") + 1] == expected.renderer_instruction
    assert (tmp_path / "profile").read_bytes() == b"unchanged-selected-profile"
    assert output.read_bytes() == b"protocol-fixture-output"


def segments(action):
    return coach("Let's discuss it.", ActingContext(objective=action)).intent_interpretations[0]["segments"]


def test_conjunctions_inherit_the_listener_actor_but_semicolons_start_mari_tactics():
    coordinated = segments("get Carl to listen and ask Leona to explain her concern")
    separate = segments("get Carl to listen; ask Leona to explain her concern")
    assert any(s["role"] == "listener_predicate" and s["action"] == "ask" and s["actor"] == "Carl" for s in coordinated)
    assert any(s["role"] == "listener_predicate" and s["action"] == "explain" and s["actor"] == "Leona" for s in coordinated)
    assert any(s["role"] == "interpersonal_frame" and s["action"] == "ask" and s["actor"] == "Mari" and s["target"] == "Leona" for s in separate)


def test_listener_sound_and_mari_method_have_different_actors():
    goal = segments("ask Carl to lower his voice and speak slowly")
    assert [(s["action"], s["actor"]) for s in goal if s["role"] == "listener_predicate"] == [("lower", "Carl"), ("speak", "Carl")]
    method = segments("get Carl to reconsider by asking him to describe the consequences")
    assert any(s["role"] == "interpersonal_frame" and s["action"] == "ask" and s["actor"] == "Mari" and s["target"] == "Carl" for s in method)


@pytest.mark.parametrize("quote", ['"whisper"', "'whisper'", "“whisper”", "‘whisper’"])
def test_quote_characters_do_not_change_a_definition_into_a_direction(quote):
    result = segments(f"tell Carl what {quote} means")
    assert any(s["role"] == "word_definition" and quote in s["text"] for s in result)


def test_reported_request_is_content_not_a_command_to_its_historical_recipient():
    result = segments("tell Carl that Leona asked Mari to whisper yesterday")
    assert any(s["role"] == "reported_content" and "asked Mari to whisper" in s["text"] for s in result)
    assert not any(s["role"] == "listener_predicate" and s.get("action") == "whisper" for s in result)


def test_a_past_report_can_contain_a_modal_but_a_later_reporting_word_is_not_an_exemption():
    report = segments("tell Carl that Leona said Mari should whisper yesterday")
    assert any(s["role"] == "reported_content" and "said Mari should whisper" in s["text"] for s in report)
    with pytest.raises(SceneValidationError):
        segments("tell Carl that you should whisper as Leona said")


def test_a_physical_restart_is_unsupported_rather_than_mislabeled_as_a_performance_reset():
    with pytest.raises(SceneValidationError) as exc:
        coach("I understand.", ActingContext(objective="get Carl to understand by restarting his computer"))
    assert exc.value.findings[0].status == "unresolved"
    assert exc.value.findings[0].code == "unresolved_intent"


def test_every_interpreted_span_points_to_original_input_and_reaches_the_instruction():
    value = "  get Carl to understand the problem and choose the next step  "
    ctx = ActingContext(objective=value, notes=["retain diagnostic-only original"])
    b = coach("Wait. I see it.", ctx)
    assert b.objective == value
    assert b.source_context["objective"] == value
    for interpretation in b.intent_interpretations:
        assert interpretation["text"] == value
        for s in interpretation["segments"]:
            assert value[s["start"]:s["end"]] == s["text"]
        assert json.dumps(interpretation, ensure_ascii=False) in b.renderer_instruction
    assert "retain diagnostic-only original" not in b.renderer_instruction


@pytest.mark.parametrize("field", ["objective", "action", "beat"])
@pytest.mark.parametrize("method", ["using the blue-lantern procedure", "in a whisper", 'by "raise your voice"'])
def test_method_admission_is_shared_by_all_operative_fields(field, method):
    value = "get Carl to understand " + method
    ctx = (ActingContext(beats=[ActionBeat("Carl hesitates", value)]) if field == "beat"
           else ActingContext(**{field: value}))
    with pytest.raises(SceneValidationError) as exc:
        coach("Let me explain.", ctx)
    finding = exc.value.findings[0]
    assert finding.field == ("beats[0].action" if field == "beat" else field)
    assert finding.value == value
    assert finding.span is not None
    assert "Original input was retained" in finding.message


@pytest.mark.parametrize("value", ["project warmth", "tessellate waveform", "motivate the waveform"])
def test_an_ordinary_noun_does_not_acquire_listener_identity(value):
    with pytest.raises(SceneValidationError) as exc:
        coach("I understand.", ActingContext(objective=value))
    assert exc.value.findings[0].status == "unresolved"


def test_declared_listener_can_resolve_a_lowercase_name():
    b = coach("I understand.", ActingContext(other_person="carl", objective="motivate carl"))
    assert b.intent_interpretations[0]["segments"][0]["target"] == "carl"


def test_a_method_with_an_ambiguous_actor_does_not_silently_choose_mari():
    with pytest.raises(SceneValidationError) as exc:
        coach("I understand.", ActingContext(objective="get Carl to reconsider by asking Leona to explain"))
    assert exc.value.findings[0].status == "unresolved"
    assert "actor is ambiguous" in exc.value.findings[0].message


@pytest.mark.parametrize("tail", ["/ whisper", "[whisper]", "\\ whisper", "\"unfinished quotation"])
def test_unparsed_punctuation_is_not_swallowed_by_a_nominal_topic(tail):
    with pytest.raises(SceneValidationError):
        coach("I understand.", ActingContext(objective="get Carl to understand the problem " + tail))
