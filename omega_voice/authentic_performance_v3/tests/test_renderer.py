import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from authentic_performance import ActingContext, coach
from omega_voice_v3 import render
from scene_contract import ActionBeat, SceneValidationError


@pytest.fixture
def engine_fixture(tmp_path):
    """An executable protocol fixture, not a speech synthesizer."""
    engine = tmp_path / "engine"
    engine.write_text("#!" + sys.executable + "\n" + '''
import json, pathlib, sys
root = pathlib.Path(__file__).parent
with (root / 'calls.jsonl').open('a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
mode = (root / 'mode').read_text()
if mode == 'fail':
    print('deliberate renderer failure')
    sys.exit(17)
if mode != 'empty':
    pathlib.Path(sys.argv[sys.argv.index('-o') + 1]).write_bytes(b'protocol-fixture-output')
''')
    engine.chmod(0o755)
    (tmp_path / "mode").write_text("success")
    model = tmp_path / "model"
    profile = tmp_path / "profile"
    model.touch()
    profile.write_bytes(b"unchanged-selected-profile")
    return dict(engine=str(engine), model=str(model), profile=str(profile),
                profile_mode="xvector", text="Wait. I see it now. Don't do it again.",
                output=str(tmp_path / "output.wav"))


@pytest.mark.parametrize("mode,flag", [("xvector", "--xvector-only"), ("graft", "--icl-only")])
def test_one_renderer_call_carries_the_complete_utterance_and_scene(engine_fixture, tmp_path, mode, flag):
    ctx = ActingContext(stakes="the only chance to reconcile",
                        beats=[ActionBeat("Carl acknowledges the mistake", "reassure Carl")])
    engine_fixture["profile_mode"] = mode
    expected = coach(engine_fixture["text"], ctx)
    result = render(**engine_fixture, context=ctx)
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 1
    args = calls[0]
    assert args[args.index("--text") + 1] == engine_fixture["text"]
    assert args[args.index("--instruct") + 1] == expected.renderer_instruction
    assert args[args.index("--load-voice") + 1] == engine_fixture["profile"]
    assert flag in args
    assert (tmp_path / "profile").read_bytes() == b"unchanged-selected-profile"
    assert pathlib.Path(result["output"]).read_bytes() == b"protocol-fixture-output"
    assert result["evidence"]["renderer_invocations"] == 1
    assert result["evidence"]["acoustic_performance"] == "not_evaluated"


@pytest.mark.parametrize("mode", ["fail", "empty"])
def test_failed_or_empty_render_has_no_fallback_and_preserves_previous_output(engine_fixture, tmp_path, mode):
    (tmp_path / "mode").write_text(mode)
    output = pathlib.Path(engine_fixture["output"])
    output.write_bytes(b"previous-output")
    with pytest.raises(RuntimeError, match="no fallback used"):
        render(**engine_fixture)
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 1
    assert output.read_bytes() == b"previous-output"


@pytest.mark.parametrize("objective", ["raise your voice", "restart the performance", "make it sparkle"])
def test_rejected_and_unresolved_intents_never_invoke_renderer(engine_fixture, tmp_path, objective):
    with pytest.raises(SceneValidationError):
        render(**engine_fixture, context=ActingContext(objective=objective))
    assert not (tmp_path / "calls.jsonl").exists()
    assert not pathlib.Path(engine_fixture["output"]).exists()


def test_invalid_profile_mode_never_invokes_renderer(engine_fixture, tmp_path):
    engine_fixture["profile_mode"] = "fallback"
    with pytest.raises(ValueError, match="profile_mode"):
        render(**engine_fixture)
    assert not (tmp_path / "calls.jsonl").exists()


@pytest.mark.parametrize("field", ["output", "brief_output"])
def test_output_cannot_overwrite_selected_profile(engine_fixture, tmp_path, field):
    engine_fixture[field] = engine_fixture["profile"]
    with pytest.raises(ValueError, match="distinct from renderer inputs"):
        render(**engine_fixture)
    assert (tmp_path / "profile").read_bytes() == b"unchanged-selected-profile"
    assert not (tmp_path / "calls.jsonl").exists()


def test_timeout_propagates_without_retry_or_replacing_output(engine_fixture, tmp_path, monkeypatch):
    calls = []
    def timed_out(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", timed_out)
    output = pathlib.Path(engine_fixture["output"])
    output.write_bytes(b"previous-output")
    with pytest.raises(subprocess.TimeoutExpired):
        render(**engine_fixture)
    assert len(calls) == 1
    assert output.read_bytes() == b"previous-output"


@pytest.mark.parametrize("context", [{"objective": "do the usual thing"}, {"unknown": "value"}])
def test_cli_errors_are_structured_and_exit_nonzero(tmp_path, context):
    ctx = tmp_path / "context.json"
    ctx.write_text(json.dumps(context))
    cp = subprocess.run([sys.executable, str(ROOT / "omega_voice_v3.py"),
                         "--engine", "/not-used", "--model", "/not-used", "--profile", "/not-used",
                         "--text", "I understand.", "--coach-only", "--context-json", str(ctx)],
                        text=True, capture_output=True)
    assert cp.returncode == 2
    assert cp.stdout == ""
    error = json.loads(cp.stderr)
    assert error["status"] == "blocked"
    assert error["findings"]
