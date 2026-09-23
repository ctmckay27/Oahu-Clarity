"""Runtime loader for the qualified Mari native prosody basis v1."""
from __future__ import annotations
import hashlib
import numpy as np
from omega_voice.native_conversational_continuity_v1.calibrated_finality import load_direction as load_finality
from .calibrated_tempo import load_direction as load_tempo

AXES=("finality_continuation","tempo")
QUALIFIED={"finality_continuation":True,"tempo":True,"onset_pressure":False,"local_focus":False}

def load_basis()->np.ndarray:
    finality=load_finality()[0]
    tempo=load_tempo()
    bank=np.stack([finality,tempo]).astype(np.float32)
    if bank.shape!=(2,29,2048) or not np.isfinite(bank).all():
        raise RuntimeError("qualified Mari prosody basis is invalid")
    return bank

def basis_sha256()->str:
    return hashlib.sha256(load_basis().astype("<f4").tobytes()).hexdigest()
