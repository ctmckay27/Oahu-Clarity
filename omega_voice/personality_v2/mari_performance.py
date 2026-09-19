from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
import re

AXES = (
    "arousal","valence","intimacy","certainty","amusement","cognitive_search",
    "restraint","fatigue","irritation","tenderness","playfulness","projection",
)

def clamp(v: float, lo: float=-1.0, hi: float=1.0) -> float:
    return max(lo, min(hi, float(v)))

@dataclass
class MariState:
    arousal: float = 0.0
    valence: float = 0.0
    intimacy: float = 0.0
    certainty: float = 0.25
    amusement: float = 0.0
    cognitive_search: float = 0.0
    restraint: float = 0.65
    fatigue: float = 0.0
    irritation: float = 0.0
    tenderness: float = 0.0
    playfulness: float = 0.0
    projection: float = 0.0

    def normalized(self) -> "MariState":
        return MariState(**{k: clamp(getattr(self,k)) for k in AXES})

    def blend(self, target: "MariState", alpha: float=0.55) -> "MariState":
        alpha=max(0.0,min(1.0,alpha))
        return MariState(**{
            k: clamp((1-alpha)*getattr(self,k)+alpha*getattr(target,k)) for k in AXES
        })

    def dict(self) -> Dict[str,float]:
        return {k:getattr(self,k) for k in AXES}

@dataclass
class InteractionContext:
    relationship: str = "familiar"
    publicness: float = 0.0
    urgency: float = 0.0
    stakes: float = 0.0
    conversational_goal: str = "respond"
    prior_state: Optional[MariState] = None
    notes: List[str] = field(default_factory=list)

@dataclass
class VocalControls:
    pace: float = 0.0
    articulation_precision: float = 0.15
    vocal_pressure: float = 0.0
    projection: float = 0.0
    attack_softness: float = 0.0
    pitch_span: float = 0.0
    pause_density: float = 0.0
    phrase_finality: float = 0.15
    rhythmic_variability: float = 0.15
    texture: float = 0.15
    breathiness: float = -0.30

    def bounded(self) -> "VocalControls":
        return VocalControls(**{k:clamp(v) for k,v in asdict(self).items()})

@dataclass
class PlannedSegment:
    index: int
    text: str
    state: MariState
    controls: VocalControls
    instruction: str
    pause_after_ms: int
    transition_reason: List[str]

@dataclass
class PerformancePlan:
    schema: str
    text: str
    initial_state: MariState
    final_state: MariState
    context: InteractionContext
    segments: List[PlannedSegment]
    laws: List[str]

    def to_dict(self) -> Dict:
        return {
            "schema": self.schema,
            "text": self.text,
            "initial_state": self.initial_state.dict(),
            "final_state": self.final_state.dict(),
            "context": {
                "relationship": self.context.relationship,
                "publicness": self.context.publicness,
                "urgency": self.context.urgency,
                "stakes": self.context.stakes,
                "conversational_goal": self.context.conversational_goal,
                "notes": self.context.notes,
            },
            "segments": [{
                "index":s.index, "text":s.text, "state":s.state.dict(),
                "controls":asdict(s.controls), "instruction":s.instruction,
                "pause_after_ms":s.pause_after_ms, "transition_reason":s.transition_reason,
            } for s in self.segments],
            "laws": list(self.laws),
        }

LAWS = [
    "same speaker identity across every segment",
    "irritation increases precision and contained force before loudness",
    "intimacy lowers projection more than it raises breathiness",
    "tenderness softens attack before increasing breathiness",
    "cognitive search and resolution are audible temporal trajectories",
    "amusement plus restraint is conveyed primarily by timing and emphasis, not a permanent vocal smile",
    "certainty reduces hesitation and strengthens phrase finality without announcer cadence",
    "no permanent affect posture; contrast across time is preserved",
    "human texture is bounded irregularity, never deliberate rasp or synthetic artifact",
    "never substitute cute/anime, generic sultry, breathy-intimate, announcer, or assistant cadence",
]

def _split_clauses(text: str) -> List[str]:
    strong=re.split(r'(?<=[.!?])\s+',text.strip())
    out: List[str]=[]
    pivot=re.compile(r'\s+(?=(?:but|because|actually|though|then|so|except|instead|still|wait|no)\b)',re.I)
    for sent in strong:
        if sent.strip():
            out.extend([p.strip() for p in pivot.split(sent) if p.strip()])
    return out or [text.strip()]

def _lexical_delta(segment: str, base: MariState) -> Tuple[MariState,List[str]]:
    d=base.dict(); why: List[str]=[]
    def add(axis: str, amount: float, reason: str):
        d[axis]=clamp(d[axis]+amount); why.append(reason)

    if re.search(r"\b(wait|hold on|hmm|maybe|perhaps|i think|not sure)\b",segment,re.I):
        add("cognitive_search",.55,"search cue"); add("certainty",-.30,"search lowers local certainty")
    if re.search(r"\b(actually|i see|right|oh|there it is|that explains)\b",segment,re.I):
        add("cognitive_search",-.40,"resolution cue"); add("certainty",.35,"resolution increases certainty")
    if re.search(r"\b(clever|funny|ridiculous|creative way|of course)\b",segment,re.I):
        add("amusement",.45,"contained amusement cue")
    if re.search(r"\b(no|wrong|do not|don't|stop|not the same|move the goalposts)\b",segment,re.I):
        add("irritation",.35,"boundary/correction cue"); add("certainty",.20,"correction increases commitment")
    if re.search(r"\b(you have enough|it is okay|we can|i will|safe|with you)\b",segment,re.I):
        add("tenderness",.25,"reassurance cue"); add("valence",.15,"warmer local stance")
    if "?" in segment: add("certainty",-.10,"question contour")
    if "!" in segment: add("arousal",.20,"exclamation raises arousal")
    return MariState(**d).normalized(),why

def _context_state(base: MariState, ctx: InteractionContext) -> MariState:
    d=base.dict()
    d["projection"] += clamp(ctx.publicness)*.45
    d["arousal"] += clamp(ctx.urgency)*.45
    d["certainty"] += clamp(ctx.stakes)*.10
    if ctx.relationship.lower() in {"close","trusted","intimate"}:
        d["intimacy"] += .35
        d["projection"] -= .18
    return MariState(**{k:clamp(v) for k,v in d.items()})

def controls_from_state(z: MariState) -> VocalControls:
    return VocalControls(
        pace=.28*z.arousal+.16*z.irritation-.28*z.fatigue-.18*z.cognitive_search,
        articulation_precision=.15+.40*z.irritation+.25*z.certainty+.12*z.arousal-.12*z.fatigue,
        vocal_pressure=.22*z.arousal+.28*z.irritation+.12*z.certainty,
        projection=z.projection-.34*z.intimacy+.10*z.arousal,
        attack_softness=.42*z.tenderness+.15*z.intimacy-.22*z.irritation,
        pitch_span=.20*z.arousal+.16*z.playfulness+.12*z.amusement-.14*z.restraint,
        pause_density=.50*z.cognitive_search+.12*z.fatigue,
        phrase_finality=.44*z.certainty+.18*z.irritation-.35*z.cognitive_search,
        rhythmic_variability=.18+.28*z.playfulness+.20*z.amusement+.22*z.cognitive_search-.12*z.fatigue,
        texture=.16+.16*abs(z.valence)+.12*z.fatigue+.10*z.amusement,
        breathiness=-.30+.06*z.intimacy+.04*z.tenderness-.08*z.irritation,
    ).bounded()

def _level(v: float, low: str, mid: str, high: str) -> str:
    if v<-.25: return low
    if v>.25: return high
    return mid

def instruction_from(z: MariState, c: VocalControls, prior: Optional[MariState], reason: List[str]) -> str:
    pieces=[
        "Same continuing adult female speaker as the loaded Mari profile.",
        _level(c.pace,"Let the pace breathe slightly.","Use natural conversational pacing.","Move a little faster with purpose, never rushed."),
        _level(c.articulation_precision,"Allow relaxed consonants without slurring.","Keep diction clear and natural, not hyperarticulated.","Give selected consonants firmer precision without announcer diction."),
        _level(c.projection,"Keep projection close and conversational.","Use ordinary conversational projection.","Project more clearly while remaining conversational, not stage-like."),
        _level(c.phrase_finality,"Leave some phrase endings locally open or provisional.","Use varied natural phrase endings.","Land committed statements cleanly and decisively."),
        "Keep habitual breathiness low; intimacy must not become whispery or sultry.",
        "Allow subtle bounded human texture and asymmetry; never add deliberate rasp, fry, or synthetic roughness.",
    ]
    if z.cognitive_search>.28:
        pieces.append("Let thought formation be audible: one brief nonuniform hesitation or reset is allowed before resolution; do not sound confused.")
    if z.amusement>.28:
        if z.restraint>.25:
            pieces.append("Let amusement appear mainly in timing and one selective emphasis, as if suppressing most of the smile; do not smile through the whole line.")
        else:
            pieces.append("Allow audible amusement with organic rhythmic lift, still as ordinary speech rather than performance acting.")
    if z.irritation>.28:
        pieces.append("Irritation should first tighten timing and articulation and add contained force; do not raise volume unless the text truly demands it.")
    if z.tenderness>.28:
        pieces.append("Warmth should soften attack and pacing before changing timbre; do not become sentimental or breathy.")
    if z.playfulness>.28:
        pieces.append("Use small rhythmic unpredictability and selective emphasis; avoid exaggerated pitch animation.")
    if prior is not None:
        dc=z.cognitive_search-prior.cognitive_search
        dcert=z.certainty-prior.certainty
        dam=z.amusement-prior.amusement
        if dc<-.25 and dcert>.18:
            pieces.append("This segment resolves a thought from the previous one: make the transition audible as recognition becoming certainty.")
        if dam>.22:
            pieces.append("Amusement enters here rather than being present from the start.")
    if reason:
        pieces.append("Local transition basis: "+", ".join(reason[:3])+".")
    pieces.append("Avoid cute/anime-girl, generic sultry, generic assistant, radio-announcer, theatrical voice-actor, and permanent-deadpan delivery.")
    return " ".join(pieces)

def pause_after(c: VocalControls, seg: str) -> int:
    base=95
    if seg.endswith("?"): base+=70
    elif seg.endswith("."): base+=35
    base += int(120*max(0.0,c.pause_density))
    base -= int(45*max(0.0,c.pace))
    return max(45,min(330,base))

def plan_performance(text: str, state: Optional[MariState]=None, context: Optional[InteractionContext]=None) -> PerformancePlan:
    if not text or not text.strip(): raise ValueError("text is required")
    context=context or InteractionContext()
    initial=(context.prior_state or state or MariState()).normalized()
    working=_context_state(initial,context)
    segments: List[PlannedSegment]=[]
    prior=initial
    for i,seg in enumerate(_split_clauses(text)):
        target,why=_lexical_delta(seg,working)
        z=prior.blend(target,alpha=.62 if why else .38)
        controls=controls_from_state(z)
        segments.append(PlannedSegment(
            i,seg,z,controls,instruction_from(z,controls,prior if i else None,why),
            pause_after(controls,seg),why
        ))
        carry=z.dict()
        carry["cognitive_search"]*=.55
        carry["arousal"]*=.78
        carry["amusement"]*=.78
        carry["irritation"]*=.82
        prior=MariState(**carry).normalized()
        working=prior
    return PerformancePlan(
        "mari-performance-plan/2.0",text,initial,
        segments[-1].state if segments else initial,context,segments,LAWS
    )
