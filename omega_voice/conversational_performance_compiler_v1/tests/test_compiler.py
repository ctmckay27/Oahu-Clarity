import copy
import re
import unittest

import numpy as np

from omega_voice.conversational_v6 import new_conversation_state, plan_turn
from omega_voice.conversational_performance_compiler_v1 import (
    append_reaction,
    capture_reaction,
    compile_native_performance,
    compile_performance_plan,
    new_learning_state,
    qualification_candidate,
    verify_native_performance_plan,
    verify_performance_plan,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [m.span() for m in re.finditer(r"\\S+", text)]
        out = {"input_ids": [3000 + i for i in range(len(spans))]}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


def make_plan(*, repair=False, low_breath=False, response=None):
    state = new_conversation_state(session_id="cpc-test", listener_id="carl")
    if low_breath:
        state["embodied"]["breath_reserve"] = 0.12

    listener = "Can you handle that today?"
    observation = {"relationship": {"familiarity": 0.95, "trust": 0.94}}
    intent = {
        "tactic": "share",
        "thought": "known",
        "certainty": 0.93,
        "objective": "answer plainly",
        "target": "say whether the task can be handled",
    }
    if repair:
        listener = "So you're saying you can't do it?"
        observation["feedback"] = "misunderstood"
        intent.update({
            "tactic": "correct",
            "objective": "repair the listener's mistaken inference",
            "target": "make the capability clear",
        })
    response = response or "Yeah, I can do that."
    return plan_turn(
        state,
        listener,
        response,
        observation=observation,
        intent=intent,
    )


def make_rich_plan(*, low_breath=False):
    state = new_conversation_state(session_id="cpc-rich", listener_id="carl")
    if low_breath:
        state["embodied"]["breath_reserve"] = 0.10
    return plan_turn(
        state,
        "Did you move the charger? I can't find it.",
        "I think I left it on the desk. Wait, no, I had it with me downstairs. Check beside the couch.",
        observation={"relationship": {"familiarity": 0.96, "trust": 0.94}},
        intent={
            "tactic": "clarify",
            "thought": "searching",
            "certainty": 0.82,
            "objective": "help Carl locate the charger while revising the answer if necessary",
            "target": "give the best current location",
        },
    )


class ConversationalPerformanceCompilerTests(unittest.TestCase):
    def test_compiles_complete_architecture(self):
        p = compile_performance_plan(make_rich_plan(low_breath=True))
        self.assertTrue(verify_performance_plan(p))
        self.assertTrue(p["thought_groups"])
        self.assertTrue(p["microtiming"]["events"])
        self.assertIn("turn_controller", p)
        self.assertIn("respiration", p)
        self.assertIn("performance_space", p)
        self.assertIn("learning_interface", p)

    def test_same_words_different_context_change_performance(self):
        ordinary = compile_performance_plan(make_plan(repair=False))
        repair = compile_performance_plan(make_plan(repair=True))
        self.assertEqual(ordinary["text"], repair["text"])
        self.assertNotEqual(ordinary["communicative_state"]["act"], repair["communicative_state"]["act"])
        self.assertNotEqual(
            ordinary["performance_space"]["channels"]["native_finality_continuation"],
            repair["performance_space"]["channels"]["native_finality_continuation"],
        )
        self.assertNotEqual(ordinary["plan_hash"], repair["plan_hash"])

    def test_relationship_is_not_direct_acoustic_recipe(self):
        p = compile_performance_plan(make_plan())
        fw = p["actuator_firewall"]
        self.assertFalse(fw["relationship_to_acoustics"])
        self.assertFalse(fw["emotion_preset_to_acoustics"])
        self.assertFalse(fw["categorical_thought_to_acoustics"])

    def test_continuous_channels_have_fixed_domain(self):
        p = compile_performance_plan(make_rich_plan())
        n = p["performance_space"]["sample_count"]
        for values in p["performance_space"]["channels"].values():
            self.assertEqual(len(values), n)
        self.assertEqual(
            p["performance_space"]["actuation"]["native_finality_continuation"],
            "QUALIFIED_MEASURED_SAME_SPEAKER_AXIS",
        )
        self.assertEqual(
            p["performance_space"]["actuation"]["pitch_movement_tendency"],
            "REPRESENTED_NOT_ACTUATED",
        )

    def test_microtiming_keeps_event_types_distinct(self):
        p = compile_performance_plan(make_rich_plan())
        kinds = {e["kind"] for e in p["microtiming"]["events"]}
        self.assertIn("RETRIEVAL_HESITATION", kinds)
        self.assertIn("SELF_CORRECTION", kinds)
        self.assertIn("THOUGHT_BOUNDARY", kinds)
        self.assertFalse(p["microtiming"]["source_withholding"])
        self.assertFalse(p["microtiming"]["generic_pause_bucket"])

    def test_respiration_is_routed_but_not_faked(self):
        p = compile_performance_plan(make_rich_plan(low_breath=True))
        r = p["respiration"]
        self.assertTrue(r["requests"])
        self.assertTrue(all(x["actuated"] is False for x in r["requests"]))
        self.assertFalse(r["source_tokens_withheld"])
        self.assertFalse(r["waveform_stitched"])
        self.assertFalse(r["fake_breath_audio"])

    def test_forbidden_respiration_mechanism_rejected(self):
        q = qualification_candidate(
            mechanism="source_token_withholding",
            identity_preserved=True,
            one_decoder_session=True,
            source_consumption_preserved=True,
            audible_timing_effect=True,
            intelligibility_preserved=True,
            evidence_id="test",
        )
        self.assertEqual(q["status"], "REJECTED")

    def test_native_adapter_preserves_identity_and_one_session(self):
        cp = make_rich_plan()
        p = compile_native_performance(cp, FakeTokenizer(), seed=77)
        self.assertTrue(verify_native_performance_plan(p))
        n = p["native_plan"]
        self.assertEqual(n["native_execution"]["decoder_sessions"], 1)
        self.assertEqual(n["native_execution"]["model_invocations"], 1)
        self.assertEqual(n["native_execution"]["unit_restarts"], 0)
        self.assertTrue(all(h["applied_frames"] == 0 for h in n["release"]["holds"]))
        self.assertEqual(n["trajectory"]["mode"], "CPC_V1_DISCOURSE_BOUNDARY_FINALITY_ONLY")
        self.assertGreater(np.count_nonzero(np.asarray(n["trajectory"]["weights"])), 0)

    def test_same_words_different_context_reach_native_trajectory(self):
        a = compile_native_performance(make_plan(repair=False), FakeTokenizer(), seed=81)
        b = compile_native_performance(make_plan(repair=True), FakeTokenizer(), seed=81)
        self.assertEqual(a["native_plan"]["text"], b["native_plan"]["text"])
        self.assertNotEqual(
            a["native_plan"]["trajectory"]["weights"],
            b["native_plan"]["trajectory"]["weights"],
        )
        self.assertEqual(
            a["native_plan"]["identity"]["profile_sha256"],
            b["native_plan"]["identity"]["profile_sha256"],
        )

    def test_learning_uses_ordinary_reaction_not_rating(self):
        p = compile_performance_plan(make_rich_plan())
        e = capture_reaction(p, "That sounded too acted and you stressed the wrong word in the second half.")
        components = {x["component"] for x in e["routes"]}
        self.assertIn("performance_projection", components)
        self.assertIn("information_focus", components)
        self.assertIn("segment_scope", components)
        self.assertFalse(e["numeric_rating_required"])
        self.assertFalse(e["automatic_parameter_update"])
        state = append_reaction(new_learning_state(), e)
        self.assertTrue(state["evidence"])
        self.assertIn("performance_projection", state["open_components"])

    def test_identity_complaint_blocks_learning_state(self):
        p = compile_performance_plan(make_plan())
        e = capture_reaction(p, "That doesn't sound like Mari anymore.")
        state = append_reaction(new_learning_state(), e)
        self.assertTrue(state["identity_blocked"])

    def test_tamper_detection(self):
        p = compile_performance_plan(make_plan())
        bad = copy.deepcopy(p)
        bad["turn_controller"]["entry_latency_s"] = 99.0
        with self.assertRaises(ValueError):
            verify_performance_plan(bad)


if __name__ == "__main__":
    unittest.main()
