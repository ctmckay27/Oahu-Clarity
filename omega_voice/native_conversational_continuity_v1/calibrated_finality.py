"""Measured Mari native finality/continuation direction, compact rank-2 projection.

Derived from MARI_ISOLATED_NATIVE_CONTROL_EVIDENCE_2026-09-19.zip.
Payload is split into small checked text parts to avoid monolithic transport corruption.
This is a same-speaker boundary-contour actuator, not an emotion/style vector.
"""
from __future__ import annotations
import base64,hashlib,zlib
from pathlib import Path
import numpy as np

SOURCE_ARCHIVE_SHA256="6f520a4937efdc1fb98fd04f956d144f999237aa4f02cfce4e41a5045d042f48"
SOURCE_NPZ_SHA256="f6fb02e3dd7117045792eb681d0f07671c2203912111b36c1ebc0d26f3ade748"
SOURCE_DIRECTION_RAW_SHA256="7759fd94fe6860494972397236d76d17636dc8124096aadb82003cfcc89a8fd2"
COMPACT_RAW_SHA256="9295ad175a3d02ae35d267ad8da6e869be2d608d1295a0cd54b3500793bbbeba"
COMPACT_B64_SHA256="b3f537bf984ee1fc6c1b1e2b8b4fb4f345eed82609a37ab42045dbba7b392055"
COSINE_TO_SOURCE=0.9797614812850952
RELATIVE_L2_ERROR=0.20016899704933167
PARTS=6

def _payload_text()->str:
    base=Path(__file__).with_name("calibrated_finality_parts")
    text="".join((base/f"part_{i:02d}.b64").read_text().strip() for i in range(PARTS))
    if hashlib.sha256(text.encode()).hexdigest()!=COMPACT_B64_SHA256:
        raise RuntimeError("calibrated direction part assembly failed integrity check")
    return text

def load_direction()->np.ndarray:
    raw=zlib.decompress(base64.b64decode(_payload_text(),validate=True))
    if hashlib.sha256(raw).hexdigest()!=COMPACT_RAW_SHA256:
        raise RuntimeError("calibrated direction compact payload failed integrity check")
    a_bytes=5*2*4
    s_bytes=2*4
    a=np.frombuffer(raw[:a_bytes],dtype="<f4").copy().reshape(5,2)
    scales=np.frombuffer(raw[a_bytes:a_bytes+s_bytes],dtype="<f4").copy()
    q=np.frombuffer(raw[a_bytes+s_bytes:],dtype=np.int8).copy().reshape(2,2048)
    active=a@(q.astype(np.float32)*scales[:,None])
    out=np.zeros((1,29,2048),dtype=np.float32)
    out[0,21:26]=active
    if not np.isfinite(out).all():
        raise RuntimeError("invalid calibrated direction reconstruction")
    return out
