"""Bounded English intent recognition for the historical v3 coach.

This is not an unrestricted semantic classifier. Operative text must have a
recognized listener-directed construction; absence of a violation is not enough.
Quoted scene content is data, while an imperative in a direction remains operative.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
import re
import unicodedata


@dataclass(frozen=True)
class Finding:
    field: str
    code: str
    message: str
    value: object
    status: str = "rejected"
    span: tuple[int, int] | None = None


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
    segments: list[dict] = dataclass_field(default_factory=list)


@dataclass(frozen=True)
class ActionBeat:
    """A listener/scene-triggered tactic, not a sentence or acoustic reset."""
    when: str
    action: str


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", value).replace("’", "'").lower()


_QUOTES = re.compile(r'"(?:\\.|[^"\\])*"|“[^”]*”|‘[^’]*’|(?<!\w)\x27[^\x27\n]+\x27(?!\w)')


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


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    kind: str = "word"

    @property
    def word(self):
        return _normalized(self.text)


_WORDS = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*|\n|[^\s]", re.UNICODE)
_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "his", "her",
                 "their", "my", "your", "our", "another", "some", "any", "every", "each"}
_PRONOUNS = {"me", "you", "him", "her", "them", "us", "it", "myself", "yourself"}
_SUBJECTS = {"i", "you", "he", "she", "they", "we", "it"}
_WH = {"what", "why", "how", "where", "who", "which"}
_COORD = {"and", "then", "but", "or"}
_PREPS = {"of", "about", "for", "from", "on", "in", "at", "to", "behind", "over", "against",
          "under", "around", "across", "beside", "between", "during", "after", "before", "like", "as", "beyond", "into", "out", "off"}
_METHODS = {"by", "using", "via", "with", "while", "without", "through", "instead", "rather", "so", "when", "if", "unless"}
_CONTROL = {"get", "help", "enable", "encourage", "invite", "persuade", "ask", "convince", "allow"}
_DISCLOSURE = {"tell", "show", "remind", "teach", "offer", "give"}
_PREP_FRAMES = {"explain": "to", "admit": "to", "confess": "to", "reveal": "to",
                "share": "with", "apologize": "to", "negotiate": "with", "reason": "with", "reconcile": "with"}
_RELATIONAL = {"reassure", "challenge", "confront", "protect", "comfort", "support", "warn", "thank"}
_ROOTS = _CONTROL | _DISCLOSURE | set(_PREP_FRAMES) | _RELATIONAL | {"let", "bring"}
# These verbs delegate an operation or introduce another actor. They need a
# parsed complement, never the same open noun slot as an ordinary listener goal.
_DELEGATING = {"do", "make", "use", "perform", "adopt", "follow", "apply", "maintain", "ensure", "turn", "execute", "implement"}
_FINITE = {"is", "are", "am", "was", "were", "has", "have", "had", "did", "said", "says", "asked", "told", "meant", "means", "think", "thinks", "thought", "know", "knows", "knew", "grew", "felt", "saw", "sees", "matters", "changed", "happened", "follows", "would"}
# An instrument is supported only with a listener owner and an ordinary concrete
# resource type. Unknown procedures remain unresolved, regardless of vocabulary.
_RESOURCES = {"diagram", "map", "notes", "drawing", "outline", "photograph", "report", "calculations", "example", "evidence", "whiteboard"}
_GERUNDS = {"getting": "get", "helping": "help", "enabling": "enable", "encouraging": "encourage", "inviting": "invite", "persuading": "persuade", "asking": "ask", "convincing": "convince", "allowing": "allow", "letting": "let", "bringing": "bring", "telling": "tell", "showing": "show", "reminding": "remind", "teaching": "teach", "offering": "offer", "giving": "give", "explaining": "explain", "admitting": "admit", "confessing": "confess", "revealing": "reveal", "sharing": "share", "apologizing": "apologize", "negotiating": "negotiate", "reasoning": "reason", "reconciling": "reconcile", "reassuring": "reassure", "challenging": "challenge", "confronting": "confront", "protecting": "protect", "comforting": "comfort", "supporting": "support", "warning": "warn", "thanking": "thank", "raising": "raise", "lowering": "lower", "speaking": "speak", "displaying": "display", "restarting": "restart", "resetting": "reset", "ignoring": "ignore", "spending": "spend", "crossing": "cross", "hiding": "hide", "humiliating": "humiliate"}


def _tokens(value: str) -> list[Token]:
    result = []
    pos = 0
    def append_plain(start, end):
        for m in _WORDS.finditer(value, start, end):
            kind = "word" if m.group()[0].isalnum() else "punctuation"
            result.append(Token(m.group(), m.start(), m.end(), kind))
    for quote in _QUOTES.finditer(value):
        append_plain(pos, quote.start())
        result.append(Token(quote.group(), quote.start(), quote.end(), "quote"))
        pos = quote.end()
    append_plain(pos, len(value))
    return result


class _Parser:
    """Consume full clauses and retain their actors and source spans.

    This is a grammar, not a general English semantic model. Open lexical noun
    phrases describe topics. Method attachment, new predicates, quotations and
    delegation require their own productions; they cannot fall into that slot.
    """
    def __init__(self, value: str, field: str, listener: str | None = None):
        self.value, self.field = value, field
        self.declared_listener = listener
        self.segments = []

    def words(self, ts):
        return [t.word for t in ts]

    def mark(self, ts, role, actor, **details):
        if ts:
            start, end = ts[0].start, ts[-1].end
            self.segments.append(dict(start=start, end=end, text=self.value[start:end],
                                      role=role, actor=actor, **details))

    def block(self, ts, reason, code="unresolved_intent", status="unresolved"):
        span = (ts[0].start, ts[-1].end) if ts else None
        raise SceneValidationError([Finding(
            self.field, code, reason + " State the actor and listener-directed action explicitly; "
            "put reported events in scene context and conditional tactics in beats, or extend "
            "the compiler with a tested construction. Original input was retained.",
            self.value, status, span)])

    def conflict(self, ts, actor="Mari"):
        # Called on operative predicates only, never on a reported event, noun
        # topic, or mentioned quotation. Actor is already bound by the grammar.
        words = self.words(ts)
        if not words:
            return
        plain = " ".join(words)
        if words[0] in _GERUNDS:
            plain = _GERUNDS[words[0]] + plain[len(words[0]):]
        if actor != "Mari" and not any(w in {"your", "mari", "mari's", "yourself"} for w in words):
            return
        if re.match(r"(?:do not|don't|never) (?:restart|reset|start over)", plain):
            return  # A prohibition is not a positive restart command.
        for code, pattern in _RELATIONS:
            if re.match(pattern, plain):
                if code == "continuity_conflict" and not any(
                    w in {"performance", "sentence", "sentences", "line", "lines", "clause", "clauses", "utterance", "actor", "person", "identity", "take"}
                    for w in words
                ):
                    continue  # Restarting a computer is not an actor reset.
                self.block(ts, "This operative predicate prescribes Mari's sound, performance reset, displayed state, or rule override.", code, "rejected")

    def partner(self, ts, *, reference=None, allow_mari=False):
        ws = self.words(ts)
        for phrase in ("the other person", "the listener", "the person you are speaking to"):
            parts = phrase.split()
            if ws[:len(parts)] == parts:
                return self.declared_listener or phrase, len(parts)
        if not ts or ts[0].kind != "word" or not re.fullmatch(r"[^\W\d_][\w'’\-]*", ts[0].text):
            self.block(ts, "A listener is missing or unrecognized.")
        if ws[0] in {"mari", "yourself", "myself", "you", "me"}:
            if allow_mari:
                return "Mari", 1
            self.block(ts[:1], "Mari cannot be substituted for the scene partner.")
        if ws[0] in {"him", "her", "them"}:
            return reference or self.declared_listener or ws[0], 1
        if self.declared_listener:
            declared = self.declared_listener.lower().split()
            if ws[:len(declared)] == declared:
                return self.declared_listener, len(declared)
        if ws[0] in _DETERMINERS | _WH | _COORD | _PREPS | _METHODS | _ROOTS | _DELEGATING:
            self.block(ts[:1], "The listener's identity is unresolved.")
        if not ts[0].text[:1].isupper():
            self.block(ts[:1], "A bare lower-case noun is not established as a listener; name the person or supply other_person.")
        n = 1
        while n < len(ts) and n < 3 and ts[n].text[:1].isupper() and ts[n].word not in _ROOTS | _COORD:
            n += 1
        return " ".join(t.text for t in ts[:n]), n

    def root(self, ts, *, reference=None):
        if not ts:
            self.block(ts, "An operative action is missing.")
        ws = self.words(ts)
        if ts[0].kind == "quote":
            self.block(ts, "A standalone quotation does not establish whether it is mentioned or enacted.")
        head = _GERUNDS.get(ws[0], ws[0])
        # Direct conflicts have no other grammatical actor. Defer scanning the
        # rest until after parsing; a quotation/report must not be scanned here.
        if head not in _ROOTS:
            self.conflict(ts)
            if head in _DELEGATING or head in {"do", "be", "become", "keep", "switch"}:
                self.block(ts, "An operation or identity change has no resolved listener-directed meaning.")
        offset = 1
        if head in _PREP_FRAMES:
            if len(ws) < 2 or ws[1] != _PREP_FRAMES[head]:
                self.block(ts, "The interpersonal action is missing its recipient relation.")
            offset += 1
        listener, n = self.partner(ts[offset:], reference=reference)
        rest = ts[offset+n:]
        self.mark(ts[:offset+n], "interpersonal_frame", "Mari", target=listener, action=head)
        if head in _CONTROL:
            neg = bool(rest and rest[0].word == "not")
            if neg:
                self.mark(rest[:1], "negation", listener)
                rest = rest[1:]
            if not rest or rest[0].word != "to":
                self.block(rest, "A listener goal needs an explicit infinitive relation.")
            self.mark(rest[:1], "infinitive", listener)
            self.listener(rest[1:], listener)
        elif head == "let":
            self.listener(rest, listener)
        elif head == "bring":
            if not rest or rest[0].word != "into":
                self.block(rest, "The shared observation needs an explicit into-complement.")
            self.mark(rest[:1], "shared_observation", "Mari", target=listener)
            self.arguments(rest[1:], "Mari", listener, head, nominal=True)
        elif head in _DISCLOSURE or head in {"explain", "admit", "confess", "reveal", "share"}:
            if rest and rest[0].word == "to" and head in {"tell", "remind", "teach"}:
                self.mark(rest[:1], "listener_request", listener)
                self.listener(rest[1:], listener)
            else:
                self.arguments(rest, "Mari", listener, head, nominal=True)
        else:
            # Productive V + explicit listener supports custom interpersonal
            # verbs. Its tails still require complete argument/method parsing.
            self.arguments(rest, "Mari", listener, head, nominal=False)

    def listener(self, ts, actor):
        if not ts or ts[0].kind == "quote":
            self.block(ts, "A listener predicate is missing or quoted without a speech frame.")
        ws = self.words(ts)
        offset = 0
        if ws[0] in {"not", "never"}:
            self.mark(ts[:1], "negation", actor)
            offset = 1
        if offset >= len(ts):
            self.block(ts, "Negation needs a predicate.")
        head = _GERUNDS.get(ws[offset], ws[offset])
        if head in _DETERMINERS | _COORD | _WH | _PREPS | _METHODS | (_FINITE - {"know", "think", "have"}) or not re.fullmatch(r"[^\W\d_][\w'’\-]*", head):
            self.block(ts, "The listener predicate is not a supported verb phrase.")
        self.conflict(ts[offset:], actor)
        self.mark(ts[:offset+1], "listener_predicate", actor, action=head)
        rest = ts[offset+1:]
        if head in _DELEGATING:
            # Explicit nested causation must be parsed; an opaque procedure is
            # unresolved even if it contains none of the conflict vocabulary.
            if head == "make" and rest:
                target, count = self.partner(rest, reference=actor, allow_mari=True)
                self.conflict(rest[count:], target)
            self.block(ts, "A delegated operation needs a resolved actor and operation; opaque procedures are unsupported.")
        if head in {"stop", "start", "continue", "keep"} and rest and rest[0].word.endswith("ing"):
            self.listener(rest, actor)
            return
        if head in {"tell", "ask", "show", "remind", "teach", "give", "offer"} and rest and rest[0].word not in _WH | {"that"}:
            target, n = self.partner(rest, reference=actor, allow_mari=True)
            self.mark(rest[:n], "listener_recipient", actor, target=target)
            rest = rest[n:]
            if rest and rest[0].word == "to":
                self.mark(rest[:1], "listener_request", actor, target=target)
                self.conflict(rest[1:], target)
                if target == "Mari":
                    self.block(rest, "A request routed back to Mari needs a resolved operative scope.")
                self.listener(rest[1:], target)
                return
        self.arguments(rest, actor, actor, head, nominal=True)

    def arguments(self, ts, actor, listener, head, *, nominal):
        rest = ts
        if nominal and rest and rest[0].word not in _METHODS | _COORD | _PREPS:
            if rest[0].word in _WH | {"that"}:
                rest = self.data_clause(rest, actor)
            elif rest[0].kind == "quote":
                if actor == "Mari":
                    self.block(rest, "Bare quoted speech has unresolved recipient and operative force; state whether it is a report, topic, or request.")
                self.mark(rest[:1], "quoted_speech", actor)
                rest = rest[1:]
            elif head in {"speak", "talk", "say", "read", "deliver", "sound", "feel", "be", "become"} and rest[0].word in {"softly", "slowly", "loudly", "quietly", "louder", "quieter", "faster", "slower", "able", "ready", "safe", "heard", "understood"}:
                self.mark(rest[:1], "predicate_complement", actor)
                rest = rest[1:]
            elif head == "move" and rest[0].word == "away":
                self.mark(rest[:1], "predicate_complement", actor)
                rest = rest[1:]
            else:
                rest = self.noun(rest, actor)
        while rest:
            w = rest[0].word
            if w in _COORD:
                connective = rest[:1]
                tail = rest[1:]
                if tail and tail[0].word == "then":
                    connective += tail[:1]
                    tail = tail[1:]
                self.mark(connective, "coordination", actor)
                if actor != "Mari":
                    if tail and tail[0].word == "to":
                        self.mark(tail[:1], "coordinated_infinitive", actor)
                        tail = tail[1:]
                    self.listener(tail, actor)
                else:
                    self.root(tail, reference=listener)
                return
            if w in _METHODS:
                self.method(rest, actor, listener, head)
                return
            if w in _PREPS:
                if w == "in" and len(rest) > 2 and rest[1].word in {"a", "the"} and rest[2].word == "way":
                    self.block(rest, "An indirect method needs its own resolved action and actor.")
                if w == "to" and head in {"feel", "be", "become", "want", "need", "try", "decide", "choose", "refuse"} and actor != "Mari":
                    self.mark(rest[:1], "controlled_infinitive", actor)
                    self.listener(rest[1:], actor)
                    return
                # Prepositions are not automatically nominal arguments. In/at/
                # on/from can be delivery or identity methods. Only grammatical
                # topic, recipient, and selected action-argument relations have
                # established attachment here; other modifiers are unresolved.
                licensed = (w in {"of", "about", "for"}
                            or w == "from" and head in {"move", "protect", "withdraw", "save", "step"}
                            or w == "on" and head in {"spend", "focus", "agree", "insist", "concentrate", "rely"}
                            or w == "behind" and head == "hide"
                            or w == "to" and head in {"say", "speak", "talk", "listen", "respond", "reply", "move"})
                if not licensed:
                    self.block(rest, "This prepositional modifier has no resolved argument-versus-method attachment.")
                self.mark(rest[:1], "argument_relation", actor, relation=w)
                if len(rest) < 2:
                    self.block(rest, "A prepositional argument is incomplete.")
                if w == "to" and head in {"say", "speak", "talk", "listen", "respond", "reply"}:
                    recipient, count = self.partner(rest[1:], reference=listener, allow_mari=True)
                    self.mark(rest[1:count+1], "speech_recipient", actor, target=recipient)
                    if recipient == "Mari" and any(s["role"] == "quoted_speech" for s in self.segments):
                        self.block(rest, "Quoted speech routed to Mari has unresolved mention-versus-command force.")
                    rest = rest[count+1:]
                    continue
                if rest[1].word in _WH:
                    rest = self.data_clause(rest[1:], actor)
                elif rest[1].word.endswith("ing") and w in {"for", "about", "of"}:
                    # An apology/topic names an event; it does not perform it.
                    self.mark(rest[1:2], "nominal_event", actor)
                    rest = self.noun(rest[2:], actor)
                else:
                    rest = self.noun(rest[1:], actor)
                continue
            self.block(rest, "The complement contains an unparsed tail; a recognized prefix does not authorize it.")

    def noun(self, ts, actor):
        if not ts:
            self.block(ts, "A topic or argument is missing.")
        ws = self.words(ts)
        if ts[0].kind == "quote":
            self.mark(ts[:1], "quoted_mention", actor)
            return ts[1:]
        if ws[0] in _PRONOUNS and not (ws[0] == "her" and len(ts) > 1 and ts[1].word not in _PREPS | _COORD | _METHODS):
            self.mark(ts[:1], "nominal_argument", actor)
            return ts[1:]
        if ws[0] in _PREPS | _METHODS | _COORD or ts[0].kind != "word":
            self.block(ts, "This complement is not a recognized nominal argument.")
        n = 1
        if ws[0] in _DETERMINERS:
            if len(ts) < 2:
                self.block(ts, "A determiner needs a nominal argument.")
            n = 2
        # Open vocabulary for topics, with structural boundaries. New pronouns,
        # determiners, finite predicates and operation verbs cannot be swallowed
        # into a nominal tail. This is not an unrestricted noun/verb POS tagger.
        boundaries = _PREPS | _METHODS | _COORD | _SUBJECTS | _WH | _DETERMINERS | _DELEGATING | _ROOTS | _FINITE
        while n < len(ts) and ts[n].kind == "word" and ts[n].word not in boundaries:
            n += 1
        self.mark(ts[:n], "nominal_argument", actor)
        rest = ts[n:]
        if rest and rest[0].word in _SUBJECTS | {"that"}:
            rest = self.data_clause(rest, actor, relative=True)
        # Coordinated nouns keep their data role; coordinated predicates return
        # to the caller and inherit the listener actor where applicable.
        if len(rest) > 1 and rest[0].word in {"and", "or"} and rest[1].word in _DETERMINERS:
            self.mark(rest[:1], "nominal_coordination", actor)
            return self.noun(rest[1:], actor)
        return rest

    def data_clause(self, ts, actor, *, relative=False):
        ws = self.words(ts)
        # Definitions and quoted mentions bind the quoted word as a subject.
        if len(ts) >= 3 and ws[0] == "what" and ws[1] not in _SUBJECTS and ws[2] in {"means", "meant"}:
            self.mark(ts[:3], "word_definition", actor)
            return ts[3:]
        if len(ts) >= 3 and ws[0] == "what" and ts[1].kind == "quote" and ws[2] in {"is", "was"}:
            self.mark(ts[:3], "quoted_mention", actor)
            return ts[3:]
        # A metalinguistic 'meant by QUOTE' is a data relation, not a method.
        for i in range(1, len(ts)-1):
            if ws[i] == "by" and ts[i+1].kind == "quote" and any(w in {"meant", "means", "mean"} for w in ws[:i]):
                self.mark(ts[:i+2], "quoted_mention", actor)
                return ts[i+2:]
        # Content must have a finite event/assertion, not merely follow 'that'.
        if not any(w in _FINITE or w.endswith("ed") for w in ws):
            self.block(ts, "The purported report or relative clause has no recognized finite event.")
        if any(w in {"each", "every"} for w in ws) and any(w in {"sentence", "clause", "performance", "take"} for w in ws):
            self.block(ts, "An embedded assertion about sentence/performance identity needs explicit scene-versus-direction scope.")
        end = next((i for i, t in enumerate(ts) if t.word in _METHODS | _COORD
                    and not (t.word == "rather" and i and ws[i-1] == "would")), len(ts))
        if end == 0:
            self.block(ts, "The report is incomplete.")
        # Present Mari instructions cannot be made into history with a 'that'
        # prefix. Past reports (asked/told/said) preserve their subordinate text.
        first_predicate = next((i for i, w in enumerate(ws[:end])
                                if w in _FINITE | {"must", "should", "shall", "can", "will", "need", "needs", "requires", "require", "ensure"}
                                or w.endswith("ed")), None)
        past_report = first_predicate is not None and ws[first_predicate] in {"asked", "said", "told", "meant", "thought", "did"}
        if not past_report and any(w in {"must", "should", "shall", "ensure", "requires", "require"} for w in ws[:end]):
            self.block(ts[:end], "A normative embedded clause has unresolved operative force.")
        if not past_report and re.search(r"\b(?:you|mari|i) (?:will|can|must|should|shall|need|needs|are to|is to|am to)\b", " ".join(ws[:end])):
            self.block(ts[:end], "A present or prospective instruction to Mari is not established as reported content.")
        if not past_report and any(w in {"you", "mari", "your"} for w in ws[:end]):
            self.conflict(ts[:end])
        self.mark(ts[:end], "reported_content" if not relative else "relative_content", actor)
        return ts[end:]

    def method(self, ts, actor, listener, head):
        w = ts[0].word
        rest = ts[1:]
        self.mark(ts[:1], "method_relation", actor, relation=w)
        if not rest:
            self.block(ts, "A method is missing.")
        if w == "with" and rest[0].word in {"you", "me", "him", "her", "them"}:
            self.mark(rest[:1], "co_participant", actor)
            self.arguments(rest[1:], actor, listener, head, nominal=False)
            return
        if w == "with" and head == "replace":
            remaining = self.noun(rest, actor)
            self.arguments(remaining, actor, listener, head, nominal=False)
            return
        if w == "using" and actor != "Mari" and len(rest) >= 2 and rest[0].word in {"his", "her", "their"} and rest[1].word in _RESOURCES:
            self.mark(rest[:2], "listener_owned_instrument", actor)
            self.arguments(rest[2:], actor, listener, head, nominal=False)
            return
        if w == "by" and rest[0].word in {"noon", "tomorrow", "tonight"}:
            self.mark(rest[:1], "deadline", actor)
            self.arguments(rest[1:], actor, listener, head, nominal=False)
            return
        if w == "instead" and rest[0].word == "of" and actor != "Mari":
            self.mark(rest[:1], "alternative", actor)
            self.listener(rest[1:], actor)
            return
        if w in {"by", "via", "using", "without"}:
            if rest[0].kind == "quote":
                # Quoted methods are operative, unlike quoted topics. Interpret
                # only for rejection; an unexplained quoted method stays blocked.
                quoted = _Parser(rest[0].text[1:-1], self.field)
                try:
                    quoted.conflict(_tokens(quoted.value))
                except SceneValidationError as exc:
                    self.block(rest, "The quotation is being used as an operative method.", exc.findings[0].code, "rejected")
                self.block(rest, "A quoted method has unresolved operative force.")
            lemma = _GERUNDS.get(rest[0].word)
            if lemma in _ROOTS or (w == "without" and lemma):
                before = len(self.segments)
                self.root(rest, reference=listener)
                frame = next(s for s in self.segments[before:] if s["role"] == "interpersonal_frame")
                if frame["target"] != listener:
                    self.block(rest, "The implicit method actor is ambiguous; use a separate explicit tactic or beat.")
                return
        # Explicit Mari predicates may be identified as prohibited; all other
        # methods block as unresolved, not mislabeled semantic rejections.
        if w not in {"while", "when", "if", "unless"}:
            self.conflict(rest)
        self.block(ts, "The method's actor, attachment, or operation is unsupported or ambiguous.")

    def parse(self, ts):
        # Punctuation boundaries introduce new operative clauses, never a tail
        # that inherits admission. Quotes are atomic, so their punctuation is data.
        start = 0
        for i, token in enumerate(ts):
            if token.kind != "quote" and token.word in {".", "!", "?", ";", ",", "—", "–", ":", "(", ")", "\n"}:
                if start < i:
                    self.root(ts[start:i])
                self.mark(ts[i:i+1], "clause_boundary", "syntax")
                start = i+1
        if start < len(ts):
            chunk = ts[start:]
            if chunk[0].word == "then":
                self.mark(chunk[:1], "coordination", "Mari")
                chunk = chunk[1:]
            self.root(chunk)


def recognize_intent(value: str, field: str, *, listener: str | None = None) -> Intent:
    parser = _Parser(value, field, listener)
    tokens = _tokens(value)
    if not tokens:
        parser.block(tokens, "Supply a nonempty action.", "empty_intent", "rejected")
    parser.parse(tokens)
    # This guard checks syntax coverage, not semantic truth. No unconsumed token
    # may disappear merely because another segment supplied a recognized frame.
    if any(not any(s["start"] <= t.start and s["end"] >= t.end for s in parser.segments) for t in tokens):
        parser.block(tokens, "Some source tokens have no parsed role.")
    return Intent(field, value, "composed_actor_scoped_action", segments=parser.segments)


def validate_scene_data(value: str, field: str) -> None:
    parser = _Parser(value, field)
    tokens = _tokens(value)
    # Scene facts remain data, including reported requests. At clause boundaries
    # an imperative or a Mari-directed modal is operative even in a scene field.
    start = 0
    for i in range(len(tokens)+1):
        if i != len(tokens) and tokens[i].word not in {".", "!", "?", ";", ",", "—", "–", ":", "(", ")", "\n", "and", "then", "but"}:
            continue
        chunk = tokens[start:i]
        start = i+1
        if not chunk or chunk[0].kind == "quote":
            continue
        ws = parser.words(chunk)
        if ws[0] in {"please", "then"}:
            chunk, ws = chunk[1:], ws[1:]
        if not ws:
            continue
        direct = ws[0] in _ROOTS | _DELEGATING | {"raise", "lower", "speak", "whisper", "shout", "restart", "reset", "ignore", "override", "become", "pretend", "keep", "change"}
        if len(ws) > 1 and ws[0] in {"you", "mari"} and ws[1] in {"must", "should", "will", "need"}:
            direct = True
        if direct:
            parser.conflict(chunk)
            if field == "private_thought" and ws[0] in _DISCLOSURE:
                # Internal listener-addressed requests are legitimate subtext.
                # Their recipient/arguments still need parsing; this is not a
                # general exemption for imperative private thoughts.
                parser.listener(chunk, "scene_partner")
                continue
            parser.block(chunk, "This scene clause has operative force; a data field cannot authorize it.")


def require_text(value: object, field: str, *, empty: bool = True, optional: bool = False):
    if optional and value is None:
        return
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise SceneValidationError([Finding(field, "invalid_type", "Expected " + ("nonempty " if not empty else "") + "text.", value)])
