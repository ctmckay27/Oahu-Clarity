"""Calibrate a candidate native speaking-tempo direction from same-Mari references.

Fast/slow waveforms are diagnostic references only. Qualification requires the
derived direction to change held-out free-generation duration while preserving
Mari identity and intelligibility.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from omega_voice.generative_v5.native import read_sequence, render
from omega_voice.generative_v5.reference_calibration import encode_reference
from .common import clean_direction, mean_activation, sha256, write_json

TEXTS = [
    "The package is ready.",
    "You left it by the window.",
    "The meeting starts at seven.",
    "That is the correct address.",
]


def tempo_reference(source: str | Path, output: str | Path, rate: float) -> dict:
    y, sr = sf.read(source, dtype="float32")
    if y.ndim != 1 or sr != 24000:
        raise ValueError("tempo calibration requires 24k mono source")
    z = librosa.effects.time_stretch(y, rate=rate)
    src_rms = max(1e-8, float(np.sqrt(np.mean(y * y))))
    dst_rms = max(1e-8, float(np.sqrt(np.mean(z * z))))
    z = z * (src_rms / dst_rms)
    peak = float(np.max(np.abs(z)))
    if peak >= 0.985:
        z *= 0.985 / peak
    sf.write(output, z, sr, subtype="PCM_16")
    return {
        "source_sha256": sha256(source),
        "audio_sha256": sha256(output),
        "rate": float(rate),
        "duration_s": len(z) / sr,
        "scope": "same-Mari diagnostic time-scale reference; never selected carrier",
    }


def run(root: str | Path) -> dict:
    root = Path(root)
    d = root / "continuation" / "prosody_basis_tempo"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    deltas = []
    neutral_means = []
    for i, text in enumerate(TEXTS):
        source = root / "continuation" / "calibration" / f"pair_{i}_statement.wav"
        if not source.is_file():
            raise FileNotFoundError(source)
        pair = {}
        for label, rate in (("fast", 1.10), ("slow", 0.90)):
            stem = d / f"{i}_{label}"
            ref = stem.with_suffix(".reference.wav")
            codes = stem.with_suffix(".codes")
            replay = stem.with_suffix(".replay.wav")
            rec = tempo_reference(source, ref, rate)
            encode_reference(root, ref, text, codes)
            render(root, text, replay, seed=93100 + i, capture=True, teacher_codes=codes)
            seq = read_sequence(replay.with_suffix(".qseq"))
            pair[label] = {
                "reference": rec,
                "replay_duration_s": sf.info(replay).duration,
                "activation": mean_activation(seq, max(0, len(seq)//10), max(1, len(seq)*9//10)),
                "capture_frames": len(seq),
            }
        ratio = pair["slow"]["replay_duration_s"] / pair["fast"]["replay_duration_s"]
        accepted = ratio >= 1.12
        if accepted:
            deltas.append(pair["fast"]["activation"] - pair["slow"]["activation"])
            neutral_means.append((pair["fast"]["activation"] + pair["slow"]["activation"]) / 2)
        rows.append({
            "pair": i,
            "text": text,
            "slow_to_fast_replay_duration_ratio": ratio,
            "accepted": accepted,
            "fast": {k:v for k,v in pair["fast"].items() if k != "activation"},
            "slow": {k:v for k,v in pair["slow"].items() if k != "activation"},
        })
        write_json(d / "RESULTS.json", rows)
    if len(deltas) < 3:
        raise RuntimeError("tempo reference contrast did not survive codec replay")
    raw = clean_direction(np.mean(deltas, axis=0), np.mean(neutral_means, axis=0))
    np.savez(d / "tempo_raw.npz", direction=raw, paired_deltas=np.stack(deltas))
    result = {
        "schema": "mari-native-tempo-calibration/1.0",
        "orientation": "positive = faster",
        "accepted_pairs": len(deltas),
        "direction_npz_sha256": sha256(d / "tempo_raw.npz"),
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
