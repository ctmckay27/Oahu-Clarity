import copy
import re
import unittest

import numpy as np

from omega_voice.conversational_v6 import new_conversation_state,plan_turn
from omega_voice.unperformed_conversation_v1 import (
    compile_unperformed_conversation,verify_unperformed_plan,
)

class FakeTokenizer:
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
        spans=[m.span() for m in re.finditer(r"\S+",text)]
        out={"input_ids":[2000+i for i in range(len(spans))]}
        if return_offsets_mapping:
            out["offset_mapping"]=spans
        return out

def sample_plan():
    state=new_conversation_state(session_id="unperformed-test",listener_id="carl")
    return plan_turn(
        state,
        "Did you move the charger? I can't find it.",
        "No, I left it on the desk. Wait, no, I had it with me downstairs. Check beside the couch.",
        observation={"relationship":{"familiarity":.95,"trust":.93}},
        intent={
            "tactic":"clarify",
            "thought":"searching",
            "certainty":.88,
            "objective":"help Carl find the charger without pretending certainty",
            "target":"give the best current location and revise it if necessary",
        },
    )

class UnperformedConversationTests(unittest.TestCase):
    def test_zeroes_injected_performance_actuation(self):
        p=compile_unperformed_conversation(sample_plan(),FakeTokenizer(),seed=17)
        self.assertTrue(verify_unperformed_plan(p))
        n=p["native_plan"]
        self.assertEqual(np.count_nonzero(np.asarray(n["trajectory"]["weights"])),0)
        self.assertEqual(np.count_nonzero(np.asarray(n["trajectory"]["onset"])),0)
        self.assertTrue(n["trajectory"]["all_injected_weights_zero"])
        self.assertEqual(n["trajectory"]["mode"],"UNPERFORMED_ZERO_INJECTED_ACTUATION")

    def test_preserves_one_session_identity_and_nonstarving_release(self):
        p=compile_unperformed_conversation(sample_plan(),FakeTokenizer())
        n=p["native_plan"]
        self.assertEqual(n["native_execution"]["decoder_sessions"],1)
        self.assertEqual(n["native_execution"]["model_invocations"],1)
        self.assertEqual(n["native_execution"]["unit_restarts"],0)
        self.assertFalse(n["laws"]["waveform_stitching"])
        self.assertTrue(all(h["applied_frames"]==0 for h in n["release"]["holds"]))
        self.assertFalse(p["identity"]["identity_changed"])

    def test_correction_survives_as_structure_not_acting(self):
        p=compile_unperformed_conversation(sample_plan(),FakeTokenizer())
        repair=[e for e in p["audibility_gate"]["events"] if e["gate"]=="LEXICAL_STRUCTURAL"]
        self.assertTrue(repair)
        self.assertTrue(any(e["source_thought"]=="correcting" or e["speech_act"]=="repair" for e in repair))
        self.assertTrue(any(e["text"].strip().lower().startswith("wait") for e in repair))
        self.assertTrue(all(e["injected_acoustic_control"] is False for e in repair))

    def test_internal_state_does_not_become_prosody_recipe(self):
        p=compile_unperformed_conversation(sample_plan(),FakeTokenizer())
        self.assertFalse(p["laws"]["categorical_state_to_prosody"])
        self.assertFalse(p["laws"]["relationship_to_performance"])
        self.assertIn("categorical_thought_trajectory",p["disabled_performance_controls"])
        self.assertIn("manual_phrase_finality",p["disabled_performance_controls"])

    def test_action_state_is_retained_as_conversation_context(self):
        p=compile_unperformed_conversation(sample_plan(),FakeTokenizer())
        a=p["action_state"]
        self.assertEqual(a["action"],"clarify")
        self.assertIn("find the charger",a["objective"])
        self.assertTrue(a["relationship_is_context_not_performance_instruction"])

    def test_deterministic(self):
        a=compile_unperformed_conversation(sample_plan(),FakeTokenizer(),seed=42)
        b=compile_unperformed_conversation(sample_plan(),FakeTokenizer(),seed=42)
        self.assertEqual(a,b)

    def test_tamper_detection(self):
        p=compile_unperformed_conversation(sample_plan(),FakeTokenizer())
        bad=copy.deepcopy(p)
        bad["native_plan"]["trajectory"]["weights"][0]=.2
        with self.assertRaises(ValueError):
            verify_unperformed_plan(bad)

if __name__=="__main__":
    unittest.main()
