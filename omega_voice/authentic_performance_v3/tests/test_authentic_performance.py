import pathlib, sys, json
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from authentic_performance import ActingContext, coach, validate_no_sound_coaching

def test_coach_directs_action_not_sound():
    b=coach("No. That's not the same problem. Look at what actually changed.")
    assert b.playable_action
    assert validate_no_sound_coaching(b.renderer_instruction)==[]

def test_other_person_is_central():
    b=coach("Tell me what happened after that.",ActingContext(other_person="Carl"))
    assert "Carl" in b.renderer_instruction
    assert "other person" not in b.other_person_focus.lower()

def test_emotion_not_assigned_as_task():
    b=coach("That's actually kind of clever, but don't do it again.")
    low=b.renderer_instruction.lower()
    assert "do not demonstrate an emotion" in low
    assert "perform a personality trait" in low

def test_single_throughline_across_sentences():
    b=coach("Wait. I see it now. That's clever. Don't do it again.")
    assert "continuous thought" in b.renderer_instruction.lower()
    assert "do not restart at sentence boundaries" in b.renderer_instruction.lower()

def test_given_circumstances_are_preserved():
    ctx=ActingContext(
        relationship="trusted",
        other_person="Carl",
        what_just_happened="Carl has been trying to solve every branch at once.",
        objective="get Carl to isolate the one blocking fact",
        obstacle="he keeps widening the problem",
        stakes="if he keeps widening it, he loses the actionable next step",
        private_thought="Stop expanding. Give me the one thing that is actually blocking you.",
    )
    b=coach("Give me the part that's actually blocking you.",ctx)
    d=b.to_dict()
    assert d["playable_action"]=="get Carl to isolate the one blocking fact"
    assert "he keeps widening the problem"==d["obstacle"]
    assert "Stop expanding" in d["private_subtext"]

def test_no_microdirection_leakage():
    b=coach("I understand why you got there, but that's not what happened.")
    forbidden=["pitch","breathy","consonant","projection","pause density","cadence","faster","slower"]
    text=b.renderer_instruction.lower()
    assert not any(x in text for x in forbidden)

def test_serializable():
    json.dumps(coach("Tell me what changed.").to_dict())
