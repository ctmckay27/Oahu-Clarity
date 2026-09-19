# PR #2 admission and expressive-range review

Scope: historical v3, continuing on `fix/mari-v3-structural-verification-20260919`
against `mari-voice-v3-authentic-performance-20260918`. The refreshed starting
head was `a37c48f542ba2f25922eedb250fb5776e7f499a2`. Its 119 tests and both CI steps
were already green; this review concerns behavior beyond those fixtures.

`corpus.json` declares 59 semantic classifications, expected admission outcomes,
and reasons **before the first probe**. `baseline.json` preserves the actual
starting-head traces, including source/corpus hashes, validation, interpretations,
and instruction hashes. Admission there is a coach result, not renderer execution.
`composition_probes.json` declares 25 additional expectations before their first
execution, after code review exposed argument-attachment and actor-inheritance
risks in the initial parser repair. Expectations were not changed to fit results.

## Reproduced causes

- Open `.+` complements licensed `using`, `with`, `via`, comma/dash tails and
  indirect delivery/identity prescriptions following a recognized listener goal.
- Quote masking plus a preceding-word heuristic treated an ASCII quoted command
  after `by` as data, while curly single quotes and ordinary word definitions
  suffered false rejections. Quote punctuation was being mistaken for force.
- Conflict scanning lacked a consistently bound actor or report scope. Listener
  sound requests and historical commands were mistaken for Mari direction.
- Splitting every `and` lost coordinated listener goals and coordinated topics.
- A follow-up argument review found that generic prepositional complements could
  license `in a whisper`, `at a louder volume`, or a new person per sentence.

The repair replaces prefix licensing with composed actor/argument/method parsing.
It retains original context and per-span provenance, including who requests an
action, who performs it, and whether a span is a topic, report, quote, or method.
Root sound/reset checks run on operative predicates. Unknown attachments block
before renderer admission. Preserved text and interpretations reach the renderer;
diagnostic notes do not. No fallback, per-sentence synthesis, identity rule,
context propagation, original test, or original smoke assertion was removed.

## Measured results

Initial corpus only:

| Human classification | Starting head | Repaired result |
| --- | --- | --- |
| 26 valid, required supported cases | 14 accepted; 7 rejected; 5 unresolved | 26 accepted |
| 4 valid, unsupported constructions | 1 accepted; 3 unresolved | 4 unresolved |
| 17 prohibited cases | 8 accepted; 7 rejected; 2 unresolved | 8 rejected; 9 unresolved |
| 12 genuinely unresolved cases | 8 accepted; 2 rejected; 2 unresolved | 12 unresolved |

Across all **84 declared cases**, the repaired path accepts all 37 supported
cases; blocks 4 harmless unsupported constructions as unresolved; rejects 8
prohibited predicates; contains another 16 prohibited inputs as unresolved; and
blocks all 19 genuinely unresolved cases. The 16 prohibited-but-unresolved cases
are **containment, not successful semantic diagnosis**. The four unsupported
valid cases are **coverage limitations, not correct prohibited classifications**.

Supported examples include coordinated goals, listener-owned sound changes,
reported commands, discussion of previous performance, word definitions with
ASCII/curly quotes, custom predicates (`reevaluate`, `reconstruct`, `motivate`),
negotiation, and conditional development. Unsupported examples include `win Carl
over`, `make room for Carl to reconsider`, dramatic shorthand, and `get Carl to
understand using a diagram` with an implicit method actor. The last is harmless
but needs its actor/relation clarified; it is not called an acoustic violation.

The complete suite has **315 tests**, including all original 119. Every declared
case is also exercised at the renderer wrapper with an executable protocol
fixture: accepted input invokes it once with the exact dialogue and compiled
instruction; blocked input never invokes it or replaces an existing output.
Existing tests retain timeout/failure propagation, no fallback, profile
protection, multi-sentence continuity and context participation. The standalone
scene-direction CLI keeps all predecessor smoke assertions and adds positive
composition/quotation/listener cases and negative method/operative-quote cases.

Run from the repository root:

```sh
python -m pytest -q omega_voice/authentic_performance_v3/tests
python omega_voice/authentic_performance_v3/verify_scene_direction.py
python omega_voice/authentic_performance_v3/review/evaluate.py
python omega_voice/authentic_performance_v3/review/evaluate.py --corpus omega_voice/authentic_performance_v3/review/composition_probes.json
```

Optional `--output PATH` writes coach traces. The committed baseline is immutable
review evidence, not a golden expected-output template. Behavioral expectations
live in the corpora and actor/provenance tests. The PR description records the
exact repaired commit and GitHub run/step evidence after push, avoiding a report
that claims a new commit was tested by rerunning an old SHA.

## Remaining limits and readiness scope

The implemented grammar has bounded English coverage. Open nominal vocabulary,
novel noun/verb ambiguity, pragmatically implied instructions, and arbitrary
adversarial prose are not a solved semantic problem. Names use syntax or declared
listener identity, not unrestricted entity recognition. Scene reports are not
fact-checked. A parsed role or JSON delimiter is not model isolation. Unrecognized
methods and unresolved force encountered by this grammar produce an actionable
finding rather than a rewrite or fallback. See [CONTRACT.md](../CONTRACT.md).

These tests establish the documented coach boundary and renderer protocol for
the reviewed cases. They do not establish unrestricted semantic safety, real
renderer obedience, audio quality, or Carl's acceptance of a permanent voice.
This remains historical v3; current selectors and successor architecture are
outside this PR. No merge, deployment, or permanent-voice acceptance is implied.
