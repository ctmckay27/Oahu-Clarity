# Historical v3 scene compilation contract, schema 1.2

## Scope and lineage

This repair targets `mari-voice-v3-authentic-performance-20260918`, based on
`dc8e6c58f550cbace6ac86d95c7600926e7f7ba8`. Run `35422249351` failed a wording
assertion at `ce07565`; run `35422267335` passed after that assertion was corrected.
Those results established coach/template behavior, not performed audio quality.

At the September 19 inspection, the newer `mari-voice-causal-runtime-20260919`
branch (refreshed at `e7e41199b50cb6a6ee20e16979e7a32437e910a4`) retains v3 unchanged and does
not import it. Its v4 README explicitly classifies v3 as historical ancestry.
The account's selected resolution state v2.4 and acting-runtime state v1.3 also
retain v3 as direction/action ancestry, not governing architecture. The dated
v0.4 machine binding in this old branch is not a current operational selector.

This PR changes only the old coach, wrapper, documentation, and verification.
It does not change G1-1, profiles, model selection, current account bindings,
the successor architecture, publication authority, or permanent voice acceptance.
The v3 prose instruction route is exercised here as a historical adapter; it
must not be imported as a replacement for the successor's structured runtime.

## Represented roles and participation

| Input or rule | Role | Destination |
| --- | --- | --- |
| Continuing Mari identity | Fixed governing rule | `identity_rule` and instruction |
| `text` | Exact dialogue, not coach directions | Brief and renderer `--text`, unchanged |
| Relationship, other person, setting | Scene / listener data | `scene_data`, instruction |
| What just happened, prior summary | Scene / history data | `scene_data`, instruction |
| Obstacle, stakes, private thought | Scene / internal data | `scene_data`, instruction |
| `objective` | Overarching listener-directed goal | Validated intent and instruction |
| `action` | Initial tactic; preset key or custom action | Validated intent and instruction |
| `beats[].when` | Scene condition for development | Preserved with the corresponding tactic |
| `beats[].action` | Conditional listener-directed tactic | Validated intent and instruction |
| `notes` | Diagnostic-only metadata | `diagnostic_notes`; never renderer direction |
| Moment-to-moment, anti-acting, continuity, simplicity rules | Fixed governing rules | Stored once in brief and emitted from those fields |
| Validation and interpretation records | Compilation evidence | Brief / receipt; not dialogue or performance direction |
| `source_context` | Exact original input, including preset keys and whitespace | Brief only; diagnostic notes do not acquire renderer authority |

The renderer also receives parsed action scopes: source spans, syntactic roles,
actors and targets. These disambiguate the validated objective/tactics without
replacing the original text. Scope records are interpretations, not execution
evidence or a sandbox. Scene and diagnostic-only records remain distinct.

The objective and action no longer silently replace each other. When only an
objective is supplied it also supplies the initial tactic for compatibility.
When neither is supplied the existing heuristic picks a library action; the
brief labels that provenance `heuristic_default`, not an observed scene fact.
Custom actions have no invented preset private motive. An explicit private
thought is preserved. Legacy preset default subtexts remain inherited choices.

Continuity means the same person and connected interaction persist. It permits
recognition, feeling, and changes of tactic. Beats are conditional developments,
not acoustic instructions, forced emotion labels, or sentence resets. A normal
utterance still makes exactly one renderer call; no per-beat synthesis is added.

## Validation mechanism and its limits

`scene_contract.py` applies an explicit role-sensitive, bounded English grammar:

1. Validate field types and reject unknown JSON keys rather than silently dropping them.
2. Keep reported scene material and quotations distinct from operative actions.
   Recognized direct override/acoustic commands in scene fields are rejected;
   recognized but unresolved imperative material is blocked.
3. Recognize actor/action/target conflict relations in operative predicates, including
   Mari-directed sound changes, performance resets, trait demonstration, and rule overrides.
   A listener action such as `ask Carl to whisper` is a different relation.
4. Parse the whole action, including complements. Constructions include listener
   goals (`get NAME to ...`, `help NAME to ...`), permission (`let NAME ...`),
   disclosure (`tell NAME ...`, `explain to NAME ...`), relational actions and
   productive `VERB + named listener` actions such as `motivate Carl`. Listener
   predicates and nominal topics use open vocabulary rather than preset moods.
   Proper-name syntax, listener-role phrases, pronouns, or `other_person` establish
   recipient identity; an arbitrary lower-case noun cannot become a listener.
5. Bind coordinated predicates to their actor. In `get Carl to listen and ask
   Leona to explain`, Carl listens and asks, and Leona explains. A semicolon or
   sentence boundary starts another Mari action. Coordinated noun topics retain
   their data role. Infinitives can repeat `to`; negation retains its scope.
6. Bind quotations to their use: definitions, topical mentions, historical
   reports, or listener speech. ASCII and curly quotation marks have the same
   treatment. A quoted method is operative; it cannot hide behind quote masking.
   A bare quotation with unresolved force blocks. Embedded current prescriptions
   cannot become past reports merely by containing a later reporting word.
7. Require an independently parsed method or argument relation. `using`, `with`,
   `via`, `by`, subordinate clauses, punctuation, and prepositional tails do not
   inherit a goal's admission. Supported methods include a Mari interpersonal
   action targeting the bound listener, an explicitly listener-owned concrete
   resource (diagram, notes, etc.), co-participation, and selected grammatical
   argument relations. Unresolved actors, opaque procedures, and ambiguous
   argument-versus-delivery attachments block before compilation/rendering.
8. Account for every source token with a parsed role and retain source spans.
   Unconsumed text returns `unresolved_intent` with the exact original field/value,
   the relevant span, and corrective advice. This token coverage check is a
   syntax invariant, not a proof of unrestricted semantic comprehension.

Examples:

| Value and role | Outcome |
| --- | --- |
| Scene: `Carl just finished his sales pitch.` | Retained and propagated |
| Scene: `Carl said, "raise your voice", and Mari disagreed.` | Quotation retained as reported content |
| Objective: `get Carl to explain his sales pitch` | Accepted within the bounded grammar |
| Action: `negotiate with Carl for another chance` | Custom action accepted |
| Objective: `get Carl to understand the problem and choose the next step` | Two predicates with Carl as actor |
| Objective: `tell Carl what ‘whisper’ means` | Mentioned word, accepted |
| Objective: `ask Carl to lower his voice and speak slowly` | Carl's actions, accepted |
| Objective: `tell Carl that Leona asked Mari to whisper yesterday` | Past request, accepted as content |
| Objective: `raise your voice at the end` | Sound-direction rejection |
| Objective: `restart the performance at every sentence` | Continuity-conflict rejection |
| Objective: `get Carl to understand me by turning up your volume` | Unresolved method; rendering blocked |
| Objective: `make it sparkle` | Unresolved intent; rendering blocked |
| Objective: `get Carl to understand using the blue-lantern procedure` | Unresolved method; rendering blocked |
| Objective: `get Carl to understand by "raise your voice"` | Operative quotation, sound-direction rejection |
| Action: `reassure Carl in a whisper` | Unresolved modifier; rendering blocked, not a claimed semantic diagnosis |
| Objective: `win Carl over` | Harmless idiom currently unsupported; unresolved, not prohibited |

Findings carry source field, original value, code, status, and corrective advice.
`SceneValidationError` is a `ValueError`; CLI validation failures emit JSON on
stderr, no success brief on stdout, and exit 2 before invoking the renderer.
There is no allow-on-unknown or automatic rewriting of rejected input.

This grammar is not unrestricted semantic understanding. Open nominal vocabulary,
the lexical noun/verb ambiguity of novel compounds, indirect pragmatic effects,
unusual English, and arbitrary adversarial prose are not exhaustively inferred.
Names use bounded syntax (up to three capitalized words, or the declared listener),
not general entity recognition. Reported assertions are not fact-checked. Methods,
idioms, phrasal verbs, and modifiers outside the productions are unsupported even
when a human can see a legitimate intent. Use the actionable unresolved finding
to clarify an actor/relation, supply a separate tactic/scene fact, or add a tested
grammar extension; do not silently paraphrase, discard, or allow the input.

`accepted_bounded` establishes the documented grammar and conflict checks.
It does not establish that arbitrary prose is harmless. A consumed nominal span
does not prove all of its lexical implications. JSON quoting, source spans and
role labels are not a sandbox that forces a language-model renderer to obey.
An application needing unrestricted semantic admission needs a stronger front
end; this historical adapter does not provide that guarantee. The concrete
prefix/tail leaks reproduced in this review are repaired, not excused by this
coverage limitation. See [the predeclared review evidence](review/README.md).

The old `validate_no_sound_coaching` function remains a **legacy lexical
diagnostic** for existing callers. It is not the admission predicate. Its word
hits may refer to legitimate scene content. The static generated-only smoke
fixture still enforces its original forbidden-word assertions.

## Verification and execution evidence

The predecessor seven tests retain their baseline and exact-template checks.
`test_scene_contract.py` checks typed positive/negative behavior and context
participation. `test_renderer.py` launches an executable protocol fixture to
observe call count, complete text/instruction transport, unchanged profile,
failure behavior, and file delivery. Timeout propagation is separately injected.
`test_review_corpus.py` runs the declared cases through the coach and executable
fixture, checks actor inheritance and source-span transport, and distinguishes
unresolved harmless cases from actual semantic rejections.
`verify_scene_direction.py` exercises the public CLI with valid and invalid
contexts without model/engine dependencies.

The wrapper validates before invocation and stages output. A failed, timed-out,
empty, or missing render cannot replace an earlier output or trigger a fallback.
An output-file receipt establishes nonempty bytes, not valid speech or perceptual
quality. The recorded command contains the temporary staging output path.

Keep these evidence levels separate:

- Coach/CLI contract: supported fixture and regression results.
- Renderer integration: one-call protocol and artifact handling.
- Acoustic performance: requires actual synthesis and qualified evaluation.
- Permanent Mari voice / publication: requires the existing acceptance authority.

Passing this workflow establishes the first two within their tested scope.
