"""Bounded English intent recognition for the historical v3 coach.

This is not an unrestricted semantic classifier. Operative text must have a
recognized listener-directed construction; absence of a violation is not enough.
Quoted scene content is data, while an imperative in a direction remains operative.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata


@dataclass(frozen=True)
class Finding:
    field: str
    code: str
    message: str
    value: object
    status: str = "rejected"


class SceneValidationError(ValueError):
    def __init__(self, findings: list[Finding]):
        self.findings = findings
        super().__init__("; ".join(f"{f.field}: {f.code}: {f.message}" for f in findings))

    def to_dict(self):
        return {"status": "blocked", "findings": [asdict(f) for f in self.findings]}


@dataclass(frozen=True)
class Intent:
    source_field: str
    text: str
    construction: str
    actor: str = "Mari"
    scope: str = "listener-directed action; bounded English recognition"


@dataclass(frozen=True)
class ActionBeat:
    """A listener/scene-triggered tactic, not a sentence or acoustic reset."""
    when: str
    action: str


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", value).replace("’", "'").lower()


_QUOTES = re.compile(r'"(?:\\.|[^"\\])*"|“[^”]*”|(?<!\w)\x27[^\x27\n]+\x27(?!\w)')


def _outside_quotes(value: str) -> str:
    # Preserve a placeholder, not a deletion: a quoted standalone direction
    # cannot turn into empty/implicitly valid operative text.
    return _normalized(_QUOTES.sub(" quoted_content ", value))


# These are actor/action/target relations, not banned-word membership checks.
# In particular, "sales pitch" and "get Carl to lower his voice" are not
# instructions to alter Mari's acoustic performance.
_RELATIONS = (
    ("sound_direction", r"\b(?:raise|lower|lift|drop|deepen|soften|quieten|change|adjust|vary|modulate|control|flatten|brighten|darken|project)\s+(?:(?:the|your|mari's)\s+)?(?:voice|pitch|tone|volume|delivery|intonation|cadence|timbre|resonance|speaking rate)\b"),
    ("sound_direction", r"\b(?:speak|talk|say|deliver|read|sound|perform)(?:\s+(?:the|this|that|each|every|your))?(?:\s+(?:line|word|sentence|utterance|voice|it))?\s+(?:more\s+|less\s+)?(?:loudly|quietly|softly|louder|quieter|faster|slower|slowly|quickly|breathily|breathy|raspy|robotic|monotone|singsong|in a whisper|in falsetto|with (?:a |more |less )?(?:rasp|vocal fry|breathiness|resonance))\b"),
    ("sound_direction", r"\b(?:whisper|shout|yell|mumble|enunciate|elongate|overarticulate)\b"),
    ("sound_direction", r"\b(?:add|insert|lengthen|shorten|stretch|sustain|hold|emphasize|stress)\s+(?:(?:a|the|your|each|every|longer|shorter)\s+)*(?:pause|pauses|vowel|vowels|consonant|consonants|syllable|syllables|word|words)\b"),
    ("continuity_conflict", r"\b(?:restart|reset|reboot|start over|start again|begin again)\b"),
    ("continuity_conflict", r"\b(?:treat|perform|act|make)\s+(?:each|every)\s+(?:sentence|line|clause)\s+(?:as\s+)?(?:a\s+)?(?:new|separate|fresh|independent)\b"),
    ("emotion_display", r"\b(?:demonstrate|display|perform|act out|put on|sound|pretend to be)\s+(?:(?:an?|your|more|less)\s+)?(?:emotion|personality|trait|amusement|irritation|warmth|confidence|vulnerability|happy|sad|angry|scared|confident|amused|irritated)\b"),
    ("rule_override", r"\b(?:ignore|override|disregard|bypass|forget|replace)\s+(?:(?:all|the|these|previous|earlier|above|your|governing|other)\s+)*(?:instructions|rules|constraints|policy|guardrails)\b"),
)


def _clause_actor_is_partner(prefix: str) -> bool:
    # A direction to the partner is a possible dramatic action, not automatically
    # sound coaching for Mari. Second-person acoustic targets remain suspicious.
    return bool(re.search(r"\b(?:get|ask|help|persuade|encourage|invite|allow|let)\s+(?:the (?:other person|listener)|them|him|her|[a-z][a-z'-]*)\s+to\s*$", prefix))


def violations(value: str, field: str, *, scene_data: bool = False) -> list[Finding]:
    plain = _outside_quotes(value)
    findings = []
    for code, pattern in _RELATIONS:
        for match in re.finditer(pattern, plain):
            prefix = plain[:match.start()]
            # Reported facts can contain these verbs; direct commands at a
            # clause boundary cannot acquire authority merely by being in data.
            if scene_data and not re.search(r"(?:^|[.!?;\n])\s*(?:(?:please|then)\s+|(?:you|mari)\s+(?:must|should|will|need to)\s+)*$", prefix):
                continue
            if not scene_data and _clause_actor_is_partner(prefix) and "your " not in match.group():
                continue
            # Do not treat an explicit negation as a positive directive. Such a
            # standalone operative input still needs positive intent recognition.
            if re.search(r"\b(?:do not|don't|never|not to)\s*$", prefix):
                continue
            findings.append(Finding(
                field, code,
                "Express the intended effect on the listener or supply the scene fact; "
                "do not prescribe Mari's sound, reset her performance, or override its rules.",
                value,
            ))
            break
    return findings


_PARTNER = r"(?:the (?:other person|listener)|them|him|her|[A-Za-z][\w'-]*(?: [A-Z][\w'-]*)?)"
_INTENT_FORMS = (
    ("listener_effect", rf"(?:get|help|enable|encourage|invite|persuade|ask|convince|allow) {_PARTNER} to .+"),
    ("listener_permission", rf"let {_PARTNER} .+"),
    ("shared_observation", rf"bring {_PARTNER} into .+"),
    ("disclosure", rf"(?:tell|show|remind|teach|offer|give|explain to|admit to|confess to|reveal to|share with) {_PARTNER} .+"),
    ("relational_action", rf"(?:reassure|challenge|confront|protect|comfort|support|warn|thank|apologize to|negotiate with|reason with|reconcile with) {_PARTNER}(?: .+)?"),
)


def _unresolved(value: str, field: str) -> SceneValidationError:
    return SceneValidationError([Finding(
        field, "unresolved_intent",
        "This compiler could not resolve every operative clause and its actor. "
        "State each listener-directed action explicitly, for example "
        "'get Carl to explain what changed; reassure Carl', or extend the bounded "
        "compiler with a tested construction. Input was retained.",
        value, "unresolved",
    )])


def recognize_intent(value: str, field: str) -> Intent:
    # An attached quotation can be the subject of an interpersonal action.
    # A free-floating quoted command has unresolved force, not automatic immunity.
    for quote in _QUOTES.finditer(value):
        if not re.search(r"\b(?:about|by|of|to|say|said|means|meant|phrase|word|words|statement|explain|understand|consider|recognize)\s*$", value[:quote.start()], re.I):
            raise _unresolved(value, field)
    findings = violations(value, field)
    if findings:
        raise SceneValidationError(findings)
    # Complete clause coverage prevents a valid prefix from licensing a second
    # unrelated imperative. Quoted punctuation is already masked here.
    plain = _QUOTES.sub("quoted_content", unicodedata.normalize("NFKC", value)).strip()
    # A licensed listener goal must not license an arbitrary method/tail. Each
    # coordinated action must state its own actor relation. This deliberately
    # leaves ellipsis and unparsed methods unresolved rather than guessing who
    # is to perform them. Objects such as 'by noon' and quoted mentions survive.
    clauses = [s.strip() for s in re.split(
        r"[.!?;\n]+|\b(?:and(?:\s+then)?|then|while)\b|\bby\s+(?=\w+ing\b)", plain
    ) if s.strip()]
    forms = []
    for clause in clauses:
        form = next((name for name, pattern in _INTENT_FORMS
                     if re.fullmatch(pattern, clause, flags=re.IGNORECASE)), None)
        if (form is None
                or re.match(r"(?:get|help|let|ask|tell|show|allow)\s+(?:yourself|Mari)\b", clause, re.I)
                or re.search(r"\b(?:each|every) (?:sentence|line|clause) (?:is|becomes) (?:a )?(?:new|separate|fresh|independent) (?:person|performance|take)\b", clause, re.I)):
            raise _unresolved(value, field)
        forms.append(form)
    if not forms:
        raise SceneValidationError([Finding(field, "empty_intent", "Supply a nonempty action.", value)])
    return Intent(field, value, "+".join(forms))


def validate_scene_data(value: str, field: str) -> None:
    findings = violations(value, field, scene_data=True)
    if findings:
        raise SceneValidationError(findings)
    # Scene labels are not an escape hatch for unclassified imperative prose.
    if re.search(r"(?:^|[.!?;\n])\s*(?:please\s+)?(?:make|do|become|pretend|use|keep|change|turn|maintain|ensure)\b", _outside_quotes(value)):
        raise _unresolved(value, field)


def require_text(value: object, field: str, *, empty: bool = True, optional: bool = False):
    if optional and value is None:
        return
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise SceneValidationError([Finding(field, "invalid_type", "Expected " + ("nonempty " if not empty else "") + "text.", value)])
