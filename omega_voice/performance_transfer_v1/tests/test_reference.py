import json,pathlib,tempfile,unittest
import numpy as np
import soundfile as sf

from omega_voice.performance_transfer_v1.reference import (
    IDENTITY_AUTHORITY,load_reference_bank,sha256,validate_performance_reference,
)

class PerformanceReferenceTests(unittest.TestCase):
    def _ref(self,td):
        p=pathlib.Path(td)/"human.wav"
        sf.write(p,np.sin(2*np.pi*180*np.arange(24000)/24000).astype("float32")*.05,24000,subtype="PCM_16")
        return {
            "audio":str(p),"text":"That should be enough for now.","sha256":sha256(p),
            "role":"human performance donor","license":"CC BY 4.0","attribution":"test donor",
            "source_url":"https://example.invalid/source.wav","donor_id":"human-1",
            "identity_authority":IDENTITY_AUTHORITY,
        }

    def test_valid_reference(self):
        with tempfile.TemporaryDirectory() as td:
            ref=self._ref(td)
            self.assertEqual(validate_performance_reference(ref),ref)

    def test_identity_authority_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            ref=self._ref(td);ref["identity_authority"]="MARI_IDENTITY"
            with self.assertRaises(ValueError): validate_performance_reference(ref)

    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            ref=self._ref(td);ref["sha256"]="0"*64
            with self.assertRaises(ValueError): validate_performance_reference(ref)

    def test_bank_duplicate_donor_fails(self):
        with tempfile.TemporaryDirectory() as td:
            ref=self._ref(td)
            bank=pathlib.Path(td)/"bank.json"
            bank.write_text(json.dumps({"schema":"mari-performance-reference-bank/1.0","references":[ref,ref]}))
            with self.assertRaises(ValueError): load_reference_bank(bank)

if __name__=="__main__": unittest.main()
