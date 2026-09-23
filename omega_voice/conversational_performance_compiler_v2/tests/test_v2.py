import re,unittest
import numpy as np
from omega_voice.conversational_v6 import new_conversation_state,plan_turn
from omega_voice.conversational_performance_compiler_v2 import compile_native_performance_v2,verify_plan_v2
from omega_voice.native_prosody_basis_v1.runtime import load_basis

class FakeTokenizer:
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
        spans=[m.span() for m in re.finditer(r"\S+",text)]
        out={"input_ids":[7000+i for i in range(len(spans))]}
        if return_offsets_mapping: out["offset_mapping"]=spans
        return out

def plan():
    s=new_conversation_state(session_id="cpc-v2-test",listener_id="carl")
    return plan_turn(
      s,"Where did you put it?",
      "I think it is upstairs. Wait, no, it is beside the couch. Check there.",
      observation={"relationship":{"familiarity":.96,"trust":.94}},
      intent={"tactic":"clarify","thought":"searching","certainty":.82,
              "objective":"repair the location answer","target":"give the corrected location"})

class Tests(unittest.TestCase):
    def test_basis_has_only_qualified_axes(self):
        b=load_basis()
        self.assertEqual(b.shape,(2,29,2048))
        self.assertTrue(np.isfinite(b).all())
    def test_v2_adds_tempo_without_rejected_axes(self):
        p=compile_native_performance_v2(plan(),FakeTokenizer(),seed=5)
        self.assertTrue(verify_plan_v2(p))
        w=np.asarray(p["trajectory"]["weights"])
        self.assertEqual(w.shape[1],2)
        self.assertGreater(np.max(np.abs(w[:,1])),0)
        self.assertEqual(p["trajectory"]["rejected_axes_not_present"],["onset_pressure","local_focus"])
        self.assertTrue(all(h["applied_frames"]==0 for h in p["source_native_plan"]["release"]["holds"]))

if __name__=="__main__": unittest.main()
