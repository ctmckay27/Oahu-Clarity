from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
import re

# This layer deliberately coaches the scene, not the sound.
FORBIDDEN_SOUND_COACHING = {
    "pitch","pitch range","f0","breathiness","breathy","consonant","articulation",
    "projection","vocal pressure","attack","phrase finality","pause density",
    "speaking rate","faster","slower","louder","quieter","rasp","fry",
    "intonation","prosody","timbre","resonance","cadence",
}

ACTION_LIBRARY = {
    "clarify": "get the other person to see the distinction you think they are missing",
    "redirect": "get the other person to stop spending attention on the wrong problem",
    "challenge": "get the other person to defend the claim instead of hiding behind wording",
    "reassure": "get the other person to feel able to take the next concrete step",
    "invite": "get the other person to tell you the thing they are holding back",
    "tease": "get the other person to recognize the absurdity with you without humiliating them",
    "set_boundary": "get the other person to stop crossing the line you just identified",
    "correct": "get the other person to replace the mistaken model with the actual one",
    "confront": "get the other person to face the fact they are avoiding",
    "protect": "get the other person to move away from the avoidable danger",
    "persuade": "get the other person to choose the course of action you think follows from the facts",
    "admit": "let the other person know the truth you would rather not have to say",
    "share": "bring the other person into an observation that matters to you",
}

@dataclass
class ActingContext:
    relationship: str = "familiar"
    other_person: str = "the person you are speaking to"
    what_just_happened: str = ""
    objective: Optional[str] = None
    obstacle: str = ""
    stakes: str = ""
    private_thought: str = ""
    action: Optional[str] = None
    prior_turn_summary: str = ""
    publicness: str = "private-conversational"
    notes: List[str] = field(default_factory=list)

@dataclass
class AuthenticPerformanceBrief:
    schema: str
    text: str
    given_circumstances: List[str]
    literal_task: str
    playable_action: str
    other_person_focus: str
    obstacle: str
    stakes: str
    private_subtext: str
    moment_to_moment_rule: str
    simplicity_rule: str
    anti_acting_rule: str
    continuity_rule: str
    renderer_instruction: str
    diagnostic_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

def _infer_action(text: str) -> str:
    t=text.lower()
    if re.search(r"\b(no|wrong|not the same|that's not|that is not)\b",t):
        return "correct"
    if re.search(r"\b(stop|don't|do not|enough|not again|line)\b",t):
        return "set_boundary"
    if re.search(r"\b(you can|we can|enough information|next step|it's okay|it is okay)\b",t):
        return "reassure"
    if re.search(r"\b(tell me|what happened|what changed|give me|show me)\b",t):
        return "clarify"
    if re.search(r"\b(clever|creative way|technically|of course|ridiculous)\b",t):
        return "tease"
    if "?" in text:
        return "invite"
    return "share"

def _literal_task(text: str) -> str:
    clauses=[x.strip() for x in re.split(r"(?<=[.!?])\s+",text.strip()) if x.strip()]
    if len(clauses)==1:
        return "say the line while pursuing one concrete effect on the listener"
    return f"carry one continuous thought through {len(clauses)} sentences without resetting the person between sentences"

def _private_subtext(ctx: ActingContext, action: str) -> str:
    if ctx.private_thought.strip():
        return ctx.private_thought.strip()
    defaults={
        "clarify":"I need you to stop giving me the explanation and give me the thing that actually happened.",
        "redirect":"I can see where your attention is going, and it is not where the problem is.",
        "challenge":"I am not going to let the wording substitute for an answer.",
        "reassure":"You can do this; I need to make the next step feel concrete rather than enormous.",
        "invite":"There is more here than you are saying, and I want you to trust me enough to say it.",
        "tease":"I see exactly what you did, and I am deciding how much I am going to let you get away with.",
        "set_boundary":"I mean this boundary, and I do not need to dramatize it to make it real.",
        "correct":"I understand why you got there, but I need you to see the actual distinction now.",
        "confront":"You already know what the uncomfortable fact is; I need you to stop stepping around it.",
        "protect":"I need you out of the avoidable risk before we discuss anything else.",
        "persuade":"The facts point somewhere; I need you to come with me rather than just understand me.",
        "admit":"I would rather this not be true, but I am not going to hide it from you.",
        "share":"I want you to notice this with me, not merely receive information from me.",
    }
    return defaults[action]

def _scene_instruction(brief: AuthenticPerformanceBrief) -> str:
    # Intentionally no acoustic engineering vocabulary.
    return (
        "Speak as the same continuing Mari. "
        f"Given circumstances: {'; '.join(brief.given_circumstances)}. "
        f"Your playable action is to {brief.playable_action}. "
        f"Keep your attention on {brief.other_person_focus}. "
        f"The obstacle is: {brief.obstacle}. "
        f"What you are not saying directly is: {brief.private_subtext}. "
        "Do not demonstrate an emotion and do not perform a personality trait. "
        "Really pursue the action in the imaginary situation, respond to the other person from moment to moment, "
        "and allow feeling to appear only as a consequence of what is happening. "
        "Keep the thought continuous. Do not restart at sentence boundaries. "
        "Use the simplest truthful choice; if nothing needs to be shown, show nothing."
    )

def validate_no_sound_coaching(instruction: str) -> List[str]:
    low=instruction.lower()
    return sorted(term for term in FORBIDDEN_SOUND_COACHING if term in low)

def coach(text: str, context: Optional[ActingContext]=None) -> AuthenticPerformanceBrief:
    if not text or not text.strip():
        raise ValueError("text is required")
    ctx=context or ActingContext()
    action=(ctx.action or _infer_action(text)).strip().lower()
    if action not in ACTION_LIBRARY:
        playable=ctx.objective.strip() if ctx.objective else action
    else:
        playable=ctx.objective.strip() if ctx.objective else ACTION_LIBRARY[action]

    circumstances=[]
    if ctx.what_just_happened.strip():
        circumstances.append(ctx.what_just_happened.strip())
    if ctx.prior_turn_summary.strip():
        circumstances.append("Immediately before this: "+ctx.prior_turn_summary.strip())
    circumstances.append("Relationship: "+ctx.relationship)
    circumstances.append("Setting: "+ctx.publicness)

    obstacle=ctx.obstacle.strip() or "the other person does not yet see or accept what you need them to"
    stakes=ctx.stakes.strip() or "the relationship and the immediate practical outcome matter more than sounding impressive"
    subtext=_private_subtext(ctx,action)

    brief=AuthenticPerformanceBrief(
        schema="mari-authentic-performance/1.0",
        text=text,
        given_circumstances=circumstances,
        literal_task=_literal_task(text),
        playable_action=playable,
        other_person_focus=ctx.other_person,
        obstacle=obstacle,
        stakes=stakes,
        private_subtext=subtext,
        moment_to_moment_rule="respond to what the other person is doing now rather than presenting a preplanned emotion",
        simplicity_rule="use the minimum performance necessary to pursue the action truthfully",
        anti_acting_rule="never demonstrate amusement, irritation, warmth, confidence, vulnerability, or personality as an effect",
        continuity_rule="one continuous person and thought across the whole utterance; sentence boundaries are linguistic, not acting resets",
        renderer_instruction="",
    )
    brief.renderer_instruction=_scene_instruction(brief)
    flags=validate_no_sound_coaching(brief.renderer_instruction)
    brief.diagnostic_flags=[f"forbidden_sound_coaching:{x}" for x in flags]
    if flags:
        raise ValueError("acting coach leaked sound-direction vocabulary: "+", ".join(flags))
    return brief
