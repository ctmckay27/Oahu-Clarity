import tempfile
import unittest
from pathlib import Path

import numpy as np

from omega_voice.native_prosody_basis_v1.common import clean_direction, orthogonalize, scale_like


class BasisMathTests(unittest.TestCase):
    def test_clean_direction_restricts_layers(self):
        rng=np.random.default_rng(7)
        delta=rng.normal(size=(29,2048)).astype("float32")
        neutral=rng.normal(size=(29,2048)).astype("float32")
        v=clean_direction(delta,neutral)
        self.assertEqual(v.shape,(29,2048))
        self.assertEqual(np.count_nonzero(v[:21]),0)
        self.assertEqual(np.count_nonzero(v[26:]),0)
        self.assertGreater(np.linalg.norm(v[21:26]),0)

    def test_orthogonalization_removes_prior_axis(self):
        rng=np.random.default_rng(8)
        a=np.zeros((29,2048),dtype="float32");a[21:26]=rng.normal(size=(5,2048))
        b=np.zeros((29,2048),dtype="float32");b[21:26]=.8*a[21:26]+rng.normal(size=(5,2048))
        q=orthogonalize(b,[a])
        cosine=float(np.sum(q*a)/(np.linalg.norm(q)*np.linalg.norm(a)))
        self.assertLess(abs(cosine),1e-5)

    def test_scale_like_matches_reference_norm(self):
        rng=np.random.default_rng(9)
        a=np.zeros((29,2048),dtype="float32");a[21:26]=rng.normal(size=(5,2048))
        b=np.zeros((29,2048),dtype="float32");b[21:26]=rng.normal(size=(5,2048))
        q,_=scale_like(a,b)
        self.assertAlmostEqual(float(np.linalg.norm(q)),float(np.linalg.norm(b)),places=3)


if __name__=="__main__":
    unittest.main()
