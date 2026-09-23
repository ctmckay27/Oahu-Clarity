"""Calibrate a candidate native local-focus direction from same-Mari references.

The reference changes only pitch excursion and level inside one measured target
word while preserving duration. Its activation difference is a calibration
hypothesis; held-out free generation must independently establish usefulness.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call
from faster_whisper import WhisperModel
from transformers import AutoTokenizer

from omega_voice.generative_v5.native import read_sequence, render
from omega_voice.generative_v5.reference_calibration import encode_reference
from .common import (
    FRAME_S, clean_direction, exact_word_alignment, focus_score, locate_target,
    mean_activation, sha256, target_token_span, write_json,
)

CASES = [
    ("The red folder is beside the window.", "red"),
    ("Please check the second drawer before you leave.", "second"),
    ("The keys are under the blue towel.", "blue"),
    ("The smaller box belongs on the upper shelf.", "upper"),
]


def focus_reference(source: str | Path, output: str | Path, start: float, end: float) -> dict:
    y, sr = sf.read(source, dtype="float64")
    if y.ndim != 1 or sr != 24000:
        raise ValueError("focus calibration requires 24k mono source")
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    manipulation = call(sound, "To Manipulation", 0.01, 70, 500)
    tier = call(manipulation, "Extract pitch tier")
    points = [
        (call(tier, "Get time from index", i), call(tier, "Get value at index", i))
        for i in range(1, call(tier, "Get number of points") + 1)
    ]
    call(tier, "Remove points between", sound.xmin, sound.xmax)
    for t, hz in points:
        u = (t - start) / max(1e-6, end - start)
        shape = math.sin(math.pi * max(0.0, min(1.0, u))) ** 2 if start <= t <= end else 0.0
        call(tier, "Add point", t, hz * 2 ** ((2.0 * shape) / 12))
    call([tier, manipulation], "Replace pitch tier")
    z = np.asarray(call(manipulation, "Get resynthesis (overlap-add)").values[0], dtype=np.float64)
    if len(z) != len(y):
        raise ValueError("focus reference changed duration")
    times = np.arange(len(y)) / sr
    u = (times - start) / max(1e-6, end - start)
    shape = np.where((times >= start) & (times <= end), np.sin(np.pi * np.clip(u, 0, 1)) ** 2, 0.0)
    z *= 10 ** ((1.5 * shape) / 20)
    peak = float(np.max(np.abs(z)))
    if peak >= 0.985:
        z *= 0.985 / peak
    sf.write(output, z, sr, subtype="PCM_16")
    return {
        "source_sha256": sha256(source),
        "audio_sha256": sha256(output),
        "target_window_s": [start, end],
        "pitch_peak_increment_st": 2.0,
        "level_peak_increment_db": 1.5,
        "duration_preserved": True,
        "scope": "same-Mari local prominence calibration stimulus only",
    }


def run(root: str | Path) -> dict:
    root = Path(root)
    d = root / "continuation" / "prosody_basis_focus"
    d.mkdir(parents=True, exist_ok=True)
    asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
    tok = AutoTokenizer.from_pretrained(root / "models" / "qwen-custom", local_files_only=True)
    rows = []
    deltas = []
    neutral_means = []
    lags = []

    for i, (text, target) in enumerate(CASES):
        source = root / "continuation" / "prosody_basis_focus_sources" / f"{i}.wav"
        if not source.is_file():
            raise FileNotFoundError(source)
        alignment = exact_word_alignment(asr, source, text)
        target_index, word = locate_target(alignment, target)
        tok_start, tok_end = target_token_span(tok, text, alignment, target_index)
        acoustic_center_frame = ((word["start"] + word["end"]) / 2) / FRAME_S
        release_frame = max(0, tok_end - 1)
        lag = acoustic_center_frame - release_frame
        lags.append(lag)

        focus = d / f"{i}_focus.reference.wav"
        neutral_codes = d / f"{i}_neutral.codes"
        focus_codes = d / f"{i}_focus.codes"
        neutral_replay = d / f"{i}_neutral.replay.wav"
        focus_replay = d / f"{i}_focus.replay.wav"

        ref = focus_reference(source, focus, word["start"], word["end"])
        encode_reference(root, source, text, neutral_codes)
        encode_reference(root, focus, text, focus_codes)
        render(root, text, neutral_replay, seed=94100+i, capture=True, teacher_codes=neutral_codes)
        render(root, text, focus_replay, seed=94100+i, capture=True, teacher_codes=focus_codes)

        nseq = read_sequence(neutral_replay.with_suffix(".qseq"))
        fseq = read_sequence(focus_replay.with_suffix(".qseq"))
        first = max(0, int(math.floor(word["start"] / FRAME_S)) - 1)
        last = max(first + 1, int(math.ceil(word["end"] / FRAME_S)) + 1)
        last = min(last, len(nseq), len(fseq))
        neutral = mean_activation(nseq, first, last)
        emphasized = mean_activation(fseq, first, last)

        na = exact_word_alignment(asr, neutral_replay, text)
        fa = exact_word_alignment(asr, focus_replay, text)
        ns = focus_score(neutral_replay, na, target_index)
        fs = focus_score(focus_replay, fa, target_index)
        delta_score = fs["coupled_focus_score"] - ns["coupled_focus_score"]
        accepted = delta_score >= 0.5
        if accepted:
            deltas.append(emphasized - neutral)
            neutral_means.append(neutral)
        rows.append({
            "case": i, "text": text, "target": target, "target_index": target_index,
            "target_token_span": [tok_start, tok_end],
            "source_to_acoustic_lag_frames": lag,
            "reference": ref,
            "neutral_replay_focus": ns,
            "focus_replay_focus": fs,
            "replay_focus_delta": delta_score,
            "accepted": accepted,
        })
        write_json(d / "RESULTS.json", rows)

    if len(deltas) < 3:
        raise RuntimeError("local focus reference contrast did not survive codec replay")
    raw = clean_direction(np.mean(deltas, axis=0), np.mean(neutral_means, axis=0))
    np.savez(d / "focus_raw.npz", direction=raw, paired_deltas=np.stack(deltas))
    result = {
        "schema": "mari-native-focus-calibration/1.0",
        "orientation": "positive = greater local coupled prominence",
        "accepted_pairs": len(deltas),
        "focus_lag_frames_median": float(np.median(lags)),
        "focus_lag_frames_all": [float(x) for x in lags],
        "direction_npz_sha256": sha256(d / "focus_raw.npz"),
        "results_sha256": sha256(d / "RESULTS.json"),
        "heldout_free_generation_qualified": False,
    }
    write_json(d / "CALIBRATION.json", result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    args = ap.parse_args()
    print(run(args.root), flush=True)


if __name__ == "__main__":
    main()
