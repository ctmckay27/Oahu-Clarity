# Historical v3 scene compilation contract, schema 1.1

## Scope and lineage

This repair targets `mari-voice-v3-authentic-performance-20260918`, based on
`dc8e6c58f550cbace6ac86d95c7600926e7f7ba8`. Run `35422249351` failed a wording
assertion at `ce07565`; run `35422267335` passed after that assertion was corrected.
Those results established coach/template behavior, not performed audio quality.

At the September 19 inspection, the newer `mari-voice-causal-runtime-20260919`
branch (`85d8c9e901064f9c5fc13b17c8cfb48781f4b016`) retains v3 unchanged and does
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
3. Recognize actor/action/target conflict relations in operative fields, including
   Mari-directed sound changes, performance resets, trait demonstration, and rule overrides.
   A listener action such as `ask Carl to whisper` is a different relation.
4. Require positive recognition of each operative clause as a listener-directed
   action. Supported constructions include `get NAME to ...`, `help NAME to ...`,
   `let NAME ...`, `tell NAME ...`, `explain to NAME ...`, `negotiate with NAME ...`,
   and other forms listed in `_INTENT_FORMS`. Names, topics, goals, and complements
   are open text, not a fixed catalog of Mari moods or actions.
5. Unrecognized actions, coordinated ellipsis, and unparsed method clauses return
   `unresolved_intent`. A recognized prefix cannot license an arbitrary second
   instruction. Write each action explicitly, e.g. `get Carl to trust you;
   explain to Carl what happened`, or extend the compiler with a tested rule.

Examples:

| Value and role | Outcome |
| --- | --- |
| Scene: `Carl just finished his sales pitch.` | Retained and propagated |
| Scene: `Carl said, "raise your voice", and Mari disagreed.` | Quotation retained as reported content |
| Objective: `get Carl to explain his sales pitch` | Accepted within the bounded grammar |
| Action: `negotiate with Carl for another chance` | Custom action accepted |
| Objective: `raise your voice at the end` | Sound-direction rejection |
| Objective: `restart the performance at every sentence` | Continuity-conflict rejection |
| Objective: `get Carl to understand me by turning up your volume` | Unresolved method; rendering blocked |
| Objective: `make it sparkle` | Unresolved intent; rendering blocked |

Findings carry source field, original value, code, status, and corrective advice.
`SceneValidationError` is a `ValueError`; CLI validation failures emit JSON on
stderr, no success brief on stdout, and exit 2 before invoking the renderer.
There is no allow-on-unknown or automatic rewriting of rejected input.

This grammar is not unrestricted semantic understanding. Open complements,
indirect implications, unusual English, and adversarial paraphrases are not
exhaustively classified. `accepted_bounded` records precisely that limited
analysis. Unsupported recognized constructions fail closed, but this is not a
proof that arbitrary prose is harmless. JSON quoting and role labels preserve
representation and reduce ambiguity; they are not a sandbox that forces a
language-model renderer to obey. Do not promote these checks into such a claim.

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
