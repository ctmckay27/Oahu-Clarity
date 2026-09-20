import copy
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from omega_voice.conversational_v6 import new_conversation_state,plan_turn
from omega_voice.conversational_realization_v1.layer import (
    RECEIPT_SCHEMA,build_realization_plan,verify_realization_plan,
)
from omega_voice.conversational_realization_v1.renderer import _join,inspect_wav


def sample_plan():
    state=new_conversation_state(session_id="test-realization",listener_id="carl")
    return plan_turn(
        state,
        "It feels better, but it still sounds like speech being rendered.",
        "Yeah. I think the missing piece is lower now. Actually, that's not precise enough. The state needs to survive into the sound itself.",
        observation={"relationship":{"familiarity":.95,"trust":.9}},
        intent={"tactic":"clarify","thought":"searching","certainty":.86},
    )


class RealizationLayerTests(unittest.TestCase):
    def test_build_preserves_identity_and_unit_order(self):
        rp=build_realization_plan(sample_plan(),base_seed=17)
        self.assertTrue(verify_realization_plan(rp))
        self.assertFalse(rp["identity"]["identity_changed"])
        self.assertEqual([u["index"] for u in rp["units"]],sorted(u["index"] for u in rp["units"]))

    def test_searching_and_correcting_are_not_same_controls(self):
        rp=build_realization_plan(sample_plan())
        searching=next(u for u in rp["units"] if u["thought"]=="searching")
        correcting=next(u for u in rp["units"] if u["thought"]=="correcting")
        self.assertLess(searching["controls"]["rate"],correcting["controls"]["rate"])
        self.assertEqual(correcting["controls"]["join"],"repair")

    def test_no_style_prompt_or_random_humanization(self):
        rp=build_realization_plan(sample_plan())
        self.assertFalse(rp["laws"]["whole_utterance_style_prompt"])
        self.assertFalse(rp["laws"]["random_humanization"])
        self.assertFalse(rp["laws"]["generic_tts_fallback"])
        self.assertFalse(rp["laws"]["fake_breath_audio"])

    def test_relationship_changes_projection_only_within_bound(self):
        p=sample_plan()
        rp=build_realization_plan(p)
        self.assertTrue(all(.92<=u["controls"]["volume"]<=1.06 for u in rp["units"]))
        self.assertGreater(rp["relationship_projection"]["intimacy"],.7)

    def test_seed_chain_is_deterministic(self):
        a=build_realization_plan(sample_plan(),base_seed=99)
        b=build_realization_plan(sample_plan(),base_seed=99)
        self.assertEqual(a,b)
        self.assertEqual([u["seed"] for u in a["units"]],[u["seed"] for u in b["units"]])

    def test_prior_receipt_continuity_is_carried(self):
        first=build_realization_plan(sample_plan())
        receipt={
            "schema":RECEIPT_SCHEMA,
            "identity":first["identity"],
            "continuity":{"terminal_token":"abc123","terminal_breath_reserve":.42},
        }
        second=build_realization_plan(sample_plan(),prior_receipt=receipt)
        self.assertEqual(second["continuity"]["prior_terminal_token"],"abc123")
        self.assertEqual(second["continuity"]["prior_terminal_breath_reserve"],.42)

    def test_wrong_identity_prior_receipt_fails_closed(self):
        bad={
            "schema":RECEIPT_SCHEMA,
            "identity":{"anchor_sha256":"wrong"},
            "continuity":{},
        }
        with self.assertRaises(ValueError):
            build_realization_plan(sample_plan(),prior_receipt=bad)

    def test_tamper_detection(self):
        rp=build_realization_plan(sample_plan())
        bad=copy.deepcopy(rp)
        bad["units"][0]["controls"]["rate"]=2.0
        with self.assertRaises(ValueError):
            verify_realization_plan(bad)

    def test_flow_boundary_crossfades(self):
        sr=24000
        a=np.sin(2*np.pi*180*np.arange(sr//2)/sr).astype("float32")*.1
        b=np.sin(2*np.pi*182*np.arange(sr//2)/sr).astype("float32")*.1
        joined,meta=_join(a,b,sr,.04,"flow")
        self.assertGreater(meta["crossfade_s"],0)
        self.assertEqual(meta["inserted_silence_s"],0)
        self.assertLess(len(joined),len(a)+len(b))

    def test_repair_boundary_preserves_pause(self):
        sr=24000
        a=np.ones(sr//4,dtype="float32")*.05
        b=np.ones(sr//4,dtype="float32")*.05
        joined,meta=_join(a,b,sr,.11,"repair")
        self.assertEqual(meta["crossfade_s"],0)
        self.assertGreaterEqual(meta["inserted_silence_s"],.10)
        self.assertGreater(len(joined),len(a)+len(b))

    def test_wav_inspection_rejects_silence(self):
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/"silence.wav"
            sf.write(p,np.zeros(24000,dtype="float32"),24000,subtype="PCM_16")
            with self.assertRaises(RuntimeError):
                inspect_wav(p)


if __name__=="__main__":
    unittest.main()
