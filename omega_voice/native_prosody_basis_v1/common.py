"""Shared machinery for identity-safe Mari native prosody-basis calibration.

Calibration stimuli may be acoustically transformed, but only activation
directions that reproduce the intended consequence during held-out free
generation with the selected Mari profile can be admitted.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import soundfile as sf

ACTIVE_LAYERS = slice(21, 26)
FRAME_S = 0.08


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower().replace("’", "'"))


def exact_word_alignment(asr: Any, path: str | Path, text: str) -> Dict[str, Any]:
    segments, _ = asr.transcribe(
        str(path), language="en", beam_size=5, temperature=0,
        condition_on_previous_text=False, vad_filter=False, word_timestamps=True,
    )
    segments = list(segments)
    hyp = " ".join(s.text.strip() for s in segments)
    words = []
    for s in segments:
        for w in s.words or []:
            tok = normalized(w.word)
            if len(tok) == 1:
                words.append({
                    "word": tok[0],
                    "start": float(w.start),
                    "end": float(w.end),
                    "probability": float(w.probability),
                })
    expected = normalized(text)
    if [w["word"] for w in words] != expected:
        raise ValueError(f"exact ASR word clock required: expected={expected!r} got={[w['word'] for w in words]!r}")
    if any(not 0 <= w["start"] < w["end"] for w in words):
        raise ValueError("invalid ASR word clock")
    return {"text": text, "asr_text": hyp, "words": words, "exact": True}


def locate_target(alignment: Dict[str, Any], target: str, occurrence: int = 0) -> Tuple[int, Dict[str, Any]]:
    target = normalized(target)[0]
    hits = [(i, w) for i, w in enumerate(alignment["words"]) if w["word"] == target]
    if not hits or occurrence >= len(hits):
        raise ValueError(f"target word not found: {target}")
    return hits[occurrence]


def target_token_span(tokenizer: Any, text: str, alignment: Dict[str, Any], target_index: int) -> Tuple[int, int]:
    matches = list(re.finditer(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", text))
    if len(matches) != len(alignment["words"]):
        raise ValueError("text/word clock token count mismatch")
    m = matches[target_index]
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(x) for x in enc["offset_mapping"]]
    hits = [i for i, (a, b) in enumerate(offsets) if b > m.start() and a < m.end()]
    if not hits:
        raise ValueError("target word has no tokenizer span")
    return hits[0], hits[-1] + 1


def focus_score(path: str | Path, alignment: Dict[str, Any], target_index: int) -> Dict[str, float]:
    import librosa

    y, sr = sf.read(path, dtype="float32")
    if y.ndim != 1 or sr != 24000:
        raise ValueError("focus score requires 24 kHz mono")
    w = alignment["words"][target_index]
    a = max(0, int(w["start"] * sr))
    b = min(len(y), max(a + 1, int(w["end"] * sr)))
    target = y[a:b]
    whole_rms = max(1e-8, float(np.sqrt(np.mean(y * y))))
    target_rms = max(1e-8, float(np.sqrt(np.mean(target * target))))
    target_level = 20 * math.log10(target_rms / whole_rms)

    f0, voiced, prob = librosa.pyin(
        y, sr=sr, fmin=70, fmax=500, frame_length=2048, hop_length=120
    )
    times = np.arange(len(f0)) * 120 / sr
    good = np.isfinite(f0) & (np.nan_to_num(prob) > 0.45)
    phrase = f0[good]
    mask = good & (times >= w["start"]) & (times <= w["end"])
    target_f0 = f0[mask]
    if len(phrase) < 5 or len(target_f0) < 2:
        pitch_rel = 0.0
    else:
        pitch_rel = float(12 * np.log2(np.median(target_f0) / np.median(phrase)))
    # A diagnostic coupled prominence score. It is not a human emphasis judgment.
    score = float(target_level + 0.35 * pitch_rel)
    return {
        "target_level_relative_db": float(target_level),
        "target_pitch_relative_st": float(pitch_rel),
        "coupled_focus_score": score,
    }


def mean_activation(seq: np.ndarray, start: int | None = None, end: int | None = None) -> np.ndarray:
    if seq.ndim != 3 or seq.shape[1:] != (29, 2048):
        raise ValueError("unexpected activation capture shape")
    lo = 0 if start is None else max(0, int(start))
    hi = len(seq) if end is None else min(len(seq), int(end))
    if hi <= lo:
        raise ValueError("empty activation window")
    return seq[lo:hi].mean(axis=0).astype(np.float32)


def clean_direction(delta: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    if delta.shape != (29, 2048) or neutral.shape != (29, 2048):
        raise ValueError("direction shape mismatch")
    v = delta.astype(np.float64).copy()
    n = neutral.astype(np.float64)
    v -= v.mean(axis=1, keepdims=True)
    denom = (n * n).sum(axis=1, keepdims=True)
    v -= n * ((v * n).sum(axis=1, keepdims=True) / np.maximum(denom, 1e-12))
    v[:21] = 0
    v[26:] = 0
    if not np.isfinite(v).all() or float(np.linalg.norm(v)) <= 1e-8:
        raise ValueError("degenerate calibrated direction")
    return v.astype(np.float32)


def orthogonalize(direction: np.ndarray, bases: Sequence[np.ndarray]) -> np.ndarray:
    v = direction.astype(np.float64).copy()
    for b in bases:
        q = b.astype(np.float64)
        denom = float(np.sum(q * q))
        if denom > 1e-12:
            v -= q * (float(np.sum(v * q)) / denom)
    v[:21] = 0
    v[26:] = 0
    if not np.isfinite(v).all() or np.linalg.norm(v) <= 1e-8:
        raise ValueError("direction collapsed during orthogonalization")
    return v.astype(np.float32)


def scale_like(direction: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, float]:
    dn = float(np.linalg.norm(direction))
    rn = float(np.linalg.norm(reference))
    if dn <= 0 or rn <= 0:
        raise ValueError("cannot scale degenerate direction")
    factor = rn / dn
    return (direction * factor).astype(np.float32), float(factor)


def write_json(path: str | Path, value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x) + "\n")
