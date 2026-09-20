import copy
import json
import pathlib
import re
import sys
import tempfile
import unittest

import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from omega_voice.conversational_v6 import new_conversation_state,plan_turn
from omega_voice.native_conversational_continuity_v1.calibrated_finality import (
    COSINE_TO_SOURCE,RELATIVE_L2_ERROR,load_direction,
)
from omega_voice.native_conversational_continuity_v1.planner import (
    compile_native_continuity_plan,materialize_packets,verify_native_continuity_plan,
)
from omega_voice.generative_v5.text_release import read_release


class FakeTokenizer:
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
        spans=[m.span() for m in re.finditer(r"\S+",text)]
        result={"input_ids":[1000+i for i in range(len(spans))]}
        if return_offsets_mapping: result["offset_mapping"]=spans
        return result


def conversation_plan():
    state=new_conversation_state(session_id="native-continuity-test",listener_id="carl")
    return plan_turn(
        state,
        "It partly worked, but I could hear the separate pieces resetting.",
        "Yeah. I think the state is finally reaching the right layer. Actually, wait. The important part is that I should stay continuous while the thought changes.",
        observation={"relationship":{"familiarity":.95,"trust":.92}},
        intent={"tactic":"clarify","thought":"searching","certainty":.84},
    )


class NativeContinuityTests(unittest.TestCase):
    def test_calibrated_direction_integrity(self):
        direction=load_direction()
        self.assertEqual(direction.shape,(1,29,2048))
        self.assertTrue(np.isfinite(direction).all())
        self.assertGreater(COSINE_TO_SOURCE,.979)
        self.assertLess(RELATIVE_L2_ERROR,.201)

    def test_one_session_invariants(self):
        plan=compile_native_continuity_plan(conversation_plan(),FakeTokenizer(),seed=17)
        self.assertTrue(verify_native_continuity_plan(plan))
        native=plan["native_execution"]
        self.assertEqual(native["decoder_sessions"],1)
        self.assertEqual(native["model_invocations"],1)
        self.assertEqual(native["unit_restarts"],0)
        self.assertEqual(native["codec_state_resets"],0)
        self.assertEqual(native["kv_cache_resets"],0)
        self.assertNotIn("seed",plan["units"][0])

    def test_release_is_monotonic_and_source_bound(self):
        plan=compile_native_continuity_plan(conversation_plan(),FakeTokenizer())
        frames=plan["release"]["earliest_frames"]
        self.assertEqual(frames[0],0)
        self.assertTrue(all(a<=b for a,b in zip(frames,frames[1:])))
        searching=[h for h in plan["release"]["holds"] if h["thought"]=="searching"]
        correcting=[h for h in plan["release"]["holds"] if h["thought"]=="correcting"]
        self.assertTrue(searching)
        self.assertTrue(correcting)
        self.assertTrue(any(h["requested_frames"]>=2 for h in searching+correcting))

    def test_native_weights_are_local_and_bounded(self):
        plan=compile_native_continuity_plan(conversation_plan(),FakeTokenizer())
        weights=np.asarray(plan["trajectory"]["weights"])
        self.assertLessEqual(float(np.max(np.abs(weights))),.30001)
        self.assertGreater(np.count_nonzero(weights),0)
        self.assertLess(np.count_nonzero(weights),len(weights))
        by_thought={s["thought"]:s["weight"] for s in plan["trajectory"]["segments"]}
        self.assertLess(by_thought["searching"],0)
        self.assertLess(by_thought["correcting"],0)

    def test_no_stitching_or_style_substitution(self):
        plan=compile_native_continuity_plan(conversation_plan(),FakeTokenizer())
        laws=plan["laws"]
        self.assertTrue(laws["one_decoder_session"])
        self.assertTrue(laws["one_continuous_codec_stream"])
        self.assertFalse(laws["separate_unit_synthesis"])
        self.assertFalse(laws["waveform_stitching"])
        self.assertFalse(laws["style_prompting"])
        self.assertFalse(laws["random_humanization"])
        self.assertTrue(laws["unsupported_dimensions_remain_unrealized"])

    def test_packets_round_trip(self):
        plan=compile_native_continuity_plan(conversation_plan(),FakeTokenizer())
        with tempfile.TemporaryDirectory() as td:
            packets=materialize_packets(plan,td)
            ids,frames=read_release(packets["release"]["path"])
            self.assertEqual(ids[:-1],plan["tokenizer"]["token_ids"])
            self.assertEqual(frames,plan["release"]["earliest_frames"])
            self.assertTrue(pathlib.Path(packets["trajectory"]["path"]).stat().st_size>1000)
            saved=json.loads(pathlib.Path(packets["plan_path"]).read_text())
            self.assertEqual(saved["plan_hash"],plan["plan_hash"])

    def test_deterministic_plan(self):
        a=compile_native_continuity_plan(conversation_plan(),FakeTokenizer(),seed=42)
        b=compile_native_continuity_plan(conversation_plan(),FakeTokenizer(),seed=42)
        self.assertEqual(a,b)

    def test_tamper_detection(self):
        plan=compile_native_continuity_plan(conversation_plan(),FakeTokenizer())
        bad=copy.deepcopy(plan)
        bad["native_execution"]["unit_restarts"]=1
        with self.assertRaises(ValueError): verify_native_continuity_plan(bad)

    def test_other_identity_fails_closed(self):
        cp=conversation_plan()
        cp["state_before"]["identity"]["profile_sha256"]="wrong"
        with self.assertRaises(ValueError): compile_native_continuity_plan(cp,FakeTokenizer())


if __name__=="__main__": unittest.main()
