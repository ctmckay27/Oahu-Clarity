import copy
import re
import unittest

from omega_voice.conversational_v6 import new_conversation_state, plan_turn
from omega_voice.conversational_performance_compiler_v1 import (
    commit_actual_delivery,
    compile_performance_plan,
    plan_native_turn,
    record_ordinary_reaction,
    verify_manifold,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        out = {"input_ids": [5000 + i for i in range(len(spans))]}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


def conversation(*, repair=False):
    state = new_conversation_state(session_id="runtime-manifold", listener_id="carl")
    observation = {"relationship": {"familiarity": 0.96, "trust": 0.94}}
    intent = {
        "tactic": "share",
        "thought": "known",
        "certainty": 0.94,
        "objective": "answer plainly",
        "target": "confirm capability",
    }
    listener = "Can you do it?"
    if repair:
        listener = "So you're saying you cannot do it?"
        observation["feedback"] = "misunderstood"
        intent["tactic"] = "correct"
        intent["objective"] = "repair a mistaken inference"
    cp = plan_turn(
        state,
        listener,
        "Yeah, I can do that.",
        observation=observation,
        intent=intent,
    )
    return state, cp, listener, observation, intent


class RuntimeManifoldTests(unittest.TestCase):
    def test_manifold_is_continuous_identity_preserving_context(self):
        _, cp, _, _, _ = conversation()
        perf = compile_performance_plan(cp)
        manifold = perf["performance_manifold"]
        self.assertTrue(verify_manifold(manifold))
        self.assertTrue(manifold["topology"]["continuous"])
        self.assertFalse(manifold["topology"]["named_presets_required"])
        self.assertTrue(manifold["identity_invariants"]["same_speaker_required"])
        self.assertFalse(manifold["identity_invariants"]["speaker_profile_substitution"])

    def test_same_words_context_moves_manifold(self):
        _, a, _, _, _ = conversation(repair=False)
        _, b, _, _, _ = conversation(repair=True)
        pa = compile_performance_plan(a)
        pb = compile_performance_plan(b)
        self.assertEqual(pa["text"], pb["text"])
        self.assertNotEqual(
            pa["performance_manifold"]["global_coordinates"],
            pb["performance_manifold"]["global_coordinates"],
        )

    def test_turn_runtime_binds_all_layers(self):
        state, _, listener, observation, intent = conversation()
        bundle = plan_native_turn(
            state,
            listener,
            "Yeah, I can do that.",
            FakeTokenizer(),
            observation=observation,
            intent=intent,
            seed=91,
        )
        self.assertEqual(bundle["delivery_status"], "PLANNED_NOT_RENDERED_NOT_DELIVERED")
        self.assertIn("performance_manifold", bundle["performance"])
        self.assertEqual(bundle["native"]["native_plan"]["native_execution"]["decoder_sessions"], 1)
        self.assertTrue(bundle["interaction_schedule"]["entry_latency_is_receiver_timing_not_waveform_silence"])

    def test_render_is_not_delivery_and_commit_is_gated(self):
        state, _, listener, observation, intent = conversation()
        bundle = plan_native_turn(
            state, listener, "Yeah, I can do that.", FakeTokenizer(),
            observation=observation, intent=intent,
        )
        with self.assertRaises(ValueError):
            commit_actual_delivery(state, bundle)
        rendered = copy.deepcopy(bundle)
        rendered["delivery_status"] = "RENDERED_NOT_DELIVERED"
        committed = commit_actual_delivery(state, rendered)
        self.assertEqual(committed["delivery_status"], "DELIVERED_AND_COMMITTED")
        self.assertEqual(committed["state_after_delivery"]["turn"], 1)
        self.assertFalse(committed["delivery_commit"]["receiver_effect_beyond_delivery_proven"])

    def test_ordinary_reaction_routes_without_numeric_homework(self):
        state, _, listener, observation, intent = conversation()
        bundle = plan_native_turn(
            state, listener, "Yeah, I can do that.", FakeTokenizer(),
            observation=observation, intent=intent,
        )
        learned = record_ordinary_reaction(bundle, "The ending sounds too acted.")
        self.assertFalse(learned["applied_to_voice"])
        self.assertFalse(learned["evidence"]["numeric_rating_required"])
        self.assertIn("performance_projection", learned["learning_state"]["open_components"])
        self.assertIn("finality_continuation", learned["learning_state"]["open_components"])


if __name__ == "__main__":
    unittest.main()
