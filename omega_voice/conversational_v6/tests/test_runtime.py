import copy
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from omega_voice.conversational_v6.runtime import (
    commit_delivery,
    interrupted_intention,
    new_conversation_state,
    observe_listener,
    plan_turn,
    verify_plan,
)


class ConversationalV6Tests(unittest.TestCase):
    def test_lexical_content_does_not_infer_listener_affect(self):
        s = new_conversation_state()
        o = observe_listener(s, "I am furious and confused about this answer.")
        self.assertEqual(o["listener_model"]["worry"], 0.0)
        self.assertEqual(o["listener_model"]["confusion"], 0.0)
        self.assertEqual(o["relationship"]["irritation"], 0.0)

    def test_explicit_listener_observation_changes_state(self):
        s = new_conversation_state()
        o = observe_listener(s, "That does not make sense.", {"confusion": .9, "feedback": "misunderstood"})
        self.assertEqual(o["listener_model"]["confusion"], .9)
        self.assertTrue(o["self_monitor"]["open_repair"])

    def test_repair_defaults_to_clarify(self):
        s = new_conversation_state()
        p = plan_turn(s, "No, that is not what I meant.", "Okay. Let me answer the narrower question.", observation={"feedback": "misunderstood"})
        self.assertEqual(p["intent"]["tactic"], "clarify")
        self.assertEqual(p["turn_taking"]["floor_strategy"], "repair")

    def test_explicit_correction_becomes_local_thought_event(self):
        s = new_conversation_state()
        p = plan_turn(s, "What happened?", "The first sensor failed. Actually, wait, the sensor was stale, not failed.")
        self.assertTrue(any(u["thought"] == "correcting" for u in p["units"]))
        self.assertTrue(any(m["kind"] == "repair_reset" for u in p["units"] for m in u["microbehavior"]))

    def test_searching_language_changes_thought_not_emotion(self):
        s = new_conversation_state()
        p = plan_turn(s, "Are you sure?", "I think the timing is the weak point, but I am not sure yet.")
        self.assertEqual(p["units"][0]["thought"], "searching")
        self.assertLessEqual(p["units"][0]["certainty"], .58)
        self.assertEqual(p["state_observed"]["relationship"]["irritation"], 0.0)

    def test_phrase_budget_splits_long_perfect_paragraph(self):
        s = new_conversation_state()
        text = " ".join(["This mechanism remains causally linked to the listener state and the current thought trajectory"] * 5) + "."
        p = plan_turn(s, "Explain it.", text)
        self.assertGreater(len(p["units"]), 2)
        self.assertTrue(all((u["end_word"] - u["start_word"]) <= p["phrase_capacity_words"] for u in p["units"]))

    def test_microbehavior_is_causal_and_deterministic(self):
        s = new_conversation_state()
        a = plan_turn(s, "Think about it.", "Maybe the second explanation is better.")
        b = plan_turn(s, "Think about it.", "Maybe the second explanation is better.")
        self.assertEqual(a, b)
        kinds = [m["kind"] for u in a["units"] for m in u["microbehavior"]]
        self.assertIn("decision_latency", kinds)
        self.assertNotIn("random_filler", kinds)

    def test_relationship_is_passed_as_causal_event_without_identity_change(self):
        s = new_conversation_state()
        p = plan_turn(s, "Tell me plainly.", "The short version is that the state changed before the display did.", observation={"relationship": {"familiarity": .95, "trust": .9}})
        rel = next(e for e in p["causal_events"] if e["id"].startswith("conv-relationship-"))
        self.assertEqual(rel["values"]["relationship.familiarity"], .95)
        self.assertEqual(p["state_before"]["identity"], p["state_observed"]["identity"])

    def test_turn_taking_latency_reflects_search_without_style_tag(self):
        s = new_conversation_state()
        fast = plan_turn(s, "What is it?", "The relay opened first.", intent={"thought": "known"})
        slow = plan_turn(s, "What is it?", "I am not sure. Let me reconstruct it.", intent={"thought": "searching"})
        self.assertGreater(slow["turn_taking"]["response_latency_s"], fast["turn_taking"]["response_latency_s"])
        self.assertFalse(slow["renderer_contract"]["style_tag_shortcut"])

    def test_commit_only_complete_delivered_units(self):
        s = new_conversation_state()
        p = plan_turn(s, "Go on.", "First point. Second point. Third point.")
        cutoff = p["units"][0]["end_word"]
        out = commit_delivery(s, p, delivered_word_count=cutoff, interrupted=True)
        self.assertEqual(len(out["self_monitor"]["spoken_units"]), 1)
        self.assertIsNotNone(interrupted_intention(out))
        self.assertGreater(len(interrupted_intention(out)["remaining_units"]), 0)

    def test_full_delivery_persists_mind_and_continuity(self):
        s = new_conversation_state()
        p = plan_turn(
            s,
            "Did the sensor fail?",
            "The display was stale. The sensor itself had not failed.",
            intent={"epistemic_proposition": "sensor remained operational", "epistemic_source": "verified incident data", "certainty": .96},
        )
        out = commit_delivery(s, p)
        self.assertEqual(out["turn"], 1)
        self.assertIn("sensor remained operational", out["mind"]["known"])
        self.assertIsNotNone(out["continuity"]["parent_state_hash"])
        self.assertIsNone(out["interaction"]["interrupted_intention"])

    def test_tamper_detection(self):
        s = new_conversation_state()
        p = plan_turn(s, "Why?", "Because the cached value stayed visible.")
        self.assertTrue(verify_plan(p))
        bad = copy.deepcopy(p)
        bad["units"][0]["text"] = "tampered"
        with self.assertRaises(ValueError):
            verify_plan(bad)

    def test_bridge_contract_with_stub_causal_runtime(self):
        fake_runtime = types.ModuleType("omega_voice.causal_v4.runtime")

        def compile_scene(text, scene, prior=None, policy=None):
            return {"text": text, "scene": scene, "prior": prior, "policy": policy, "renderer_instruction": None}

        fake_runtime.compile_scene = compile_scene
        fake_pkg = types.ModuleType("omega_voice.causal_v4")
        sys.modules["omega_voice.causal_v4"] = fake_pkg
        sys.modules["omega_voice.causal_v4.runtime"] = fake_runtime
        try:
            from omega_voice.conversational_v6.bridge import compile_causal
            s = new_conversation_state()
            p = plan_turn(s, "What changed?", "The state changed before the display updated.", observation={"confusion": .2})
            c = compile_causal(p)
            self.assertEqual(c["policy"], "listener_causal_v6")
            self.assertEqual(c["text"], p["response_text"])
            self.assertIsNone(c["renderer_instruction"])
            self.assertFalse(c["scene"]["metadata"]["renderer_contract"]["generic_tts_fallback"])
        finally:
            sys.modules.pop("omega_voice.causal_v4.runtime", None)
            sys.modules.pop("omega_voice.causal_v4", None)


if __name__ == "__main__":
    unittest.main()
