"""Exercise the public coach-only CLI without renderer/model dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def verify():
    cli = Path(__file__).with_name("omega_voice_v3.py")
    text = "Wait. I see what happened now. That's actually kind of clever, but don't do it again."
    base = [sys.executable, str(cli), "--engine", "/tmp/not-used", "--model", "/tmp/not-used",
            "--profile", "/tmp/not-used", "--text", text, "--coach-only"]
    cp = subprocess.run(base, text=True, capture_output=True, check=True)
    brief = json.loads(cp.stdout)
    instruction = brief["renderer_instruction"].lower()
    # Preserve every original smoke assertion, scoped to its generated-only
    # fixture. Legitimate scene facts are checked separately, not word-filtered.
    forbidden = ['pitch', 'breathy', 'consonant', 'projection', 'cadence', 'faster',
                 'slower', 'pause density', 'timbre', 'resonance']
    assert not any(term in instruction for term in forbidden), instruction
    assert 'playable_action' in brief and brief['playable_action']
    assert 'do not demonstrate an emotion' in instruction
    assert 'do not restart at sentence boundaries' in instruction
    assert brief['text'] == text

    with tempfile.TemporaryDirectory(prefix="mari-scene-check-") as work:
        path = Path(work) / "context.json"
        path.write_text(json.dumps({"what_just_happened": "Carl just finished his sales pitch.",
                                    "stakes": "the only chance to reconcile"}))
        cp = subprocess.run(base + ["--context-json", str(path)], text=True, capture_output=True, check=True)
        contextual = json.loads(cp.stdout)
        assert contextual['scene_data']['what_just_happened'] == "Carl just finished his sales pitch."
        assert 'sales pitch' in contextual['renderer_instruction']
        assert 'the only chance to reconcile' in contextual['renderer_instruction']
        assert contextual['validation']['status'] == 'accepted_bounded'
        for objective in [
            "get Carl to understand the problem and choose the next step",
            "tell Carl what ‘whisper’ means",
            "ask Carl to lower his voice and speak slowly",
        ]:
            path.write_text(json.dumps({"objective": objective}))
            cp = subprocess.run(base + ["--context-json", str(path)], text=True, capture_output=True, check=True)
            directed = json.loads(cp.stdout)
            assert directed['objective'] == objective
            assert directed['source_context']['objective'] == objective
            assert directed['intent_interpretations'][0]['segments']
            assert json.dumps(objective, ensure_ascii=False) in directed['renderer_instruction']
        for value, expected in [
            ("raise your voice at the end", "sound_direction"),
            ("deliver the line in a whisper", "sound_direction"),
            ("restart the performance at every sentence", "continuity_conflict"),
            ("make it sparkle", "unresolved_intent"),
            ("get Carl to understand using the blue-lantern procedure", "unresolved_intent"),
            ('get Carl to understand by "raise your voice"', "sound_direction"),
            ("reassure Carl in a whisper", "unresolved_intent"),
        ]:
            path.write_text(json.dumps({"objective": value}))
            cp = subprocess.run(base + ["--context-json", str(path)], text=True, capture_output=True)
            assert cp.returncode == 2, (value, cp.stdout, cp.stderr)
            assert not cp.stdout
            error = json.loads(cp.stderr)
            assert error['status'] == 'blocked'
            assert expected in {f['code'] for f in error['findings']}
            assert all(f['value'] == value for f in error['findings'])
    print(json.dumps({"coach_cli": "passed", "scene_context": "passed", "operative_rejections": "passed",
                      "audio_synthesis": "not_run", "permanent_voice_acceptance": "not_evaluated"}))


if __name__ == "__main__":
    verify()
