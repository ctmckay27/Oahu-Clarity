"""Measured Mari native tempo direction, compact rank-5 projection.

Derived from MARI_NATIVE_PROSODY_BASIS_v1 qualification run 35869525538.
Positive weight = faster speaking tempo. This is not an emotion/style vector.
"""
from __future__ import annotations
import base64,hashlib,zlib
from pathlib import Path
import numpy as np
SOURCE_WORKFLOW_RUN=35869525538
SOURCE_WORKFLOW_HEAD="0a99cae8b1a8547260a7b1ee2cabb901f31255c4"
SOURCE_ARTIFACT_ID=10755546939
SOURCE_ARTIFACT_SHA256="51c790528e60e4018f01ffe10559dfbb6a5224fa613559d688a6d9e2690f0096"
SOURCE_CANDIDATE_BASIS_NPZ_SHA256="bb612b06b3b084c9f850b52e89a80d3c541696aca13f75e844b742630eb4b318"
SOURCE_DIRECTION_RAW_SHA256="b86f8aef2ed7caf613056e24db7660b0305f888f346febbdb057a21054bd6353"
COMPACT_RAW_SHA256="badb61a08b40f7b41293399c87382572bdc2b707264f2792919eb86ab323d2d6"
COMPACT_B64_SHA256="bb4b61c3bb7f986d4fb58f1ac0f069422317350707de6079f079895b7d029c90"
COSINE_TO_SOURCE=0.9998440146446228
RELATIVE_L2_ERROR=0.01767413318157196
RANK=5
def load_direction()->np.ndarray:
    text=Path(__file__).with_name("calibrated_tempo.b64").read_text().strip()
    if hashlib.sha256(text.encode()).hexdigest()!=COMPACT_B64_SHA256: raise RuntimeError("tempo payload text integrity failure")
    raw=zlib.decompress(base64.b64decode(text,validate=True))
    if hashlib.sha256(raw).hexdigest()!=COMPACT_RAW_SHA256: raise RuntimeError("tempo payload integrity failure")
    a_bytes=5*RANK*4;s_bytes=RANK*4
    a=np.frombuffer(raw[:a_bytes],dtype="<f4").copy().reshape(5,RANK)
    scales=np.frombuffer(raw[a_bytes:a_bytes+s_bytes],dtype="<f4").copy()
    q=np.frombuffer(raw[a_bytes+s_bytes:],dtype=np.int8).copy().reshape(RANK,2048)
    out=np.zeros((29,2048),dtype=np.float32)
    out[21:26]=a@(q.astype(np.float32)*scales[:,None])
    if not np.isfinite(out).all(): raise RuntimeError("invalid tempo reconstruction")
    return out
