"""Held-out free-generation qualification for Mari native prosody basis v1.

Calibration directions are admitted only if their predicted acoustic consequence
survives free generation while preserving intelligibility and Mari speaker
similarity. No perceptual-naturalness claim is made here.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from transformers import AutoTokenizer

from omega_voice.generative_v5.attack_calibration import attack_measure
from omega_voice.generative_v5.native import render, write_trajectory
from omega_voice.generative_v5.text_release import write_release
from omega_voice.native_conversational_continuity_v1.calibrated_finality import load_direction
from .common import (
    exact_word_alignment, focus_score, locate_target, normalized, orthogonalize,
    scale_like, sha256, target_token_span, write_json,
)

AXES = ["finality_continuation", "onset_pressure", "tempo", "local_focus"]


def edit_distance(a: List[str], b: List[str]) -> int:
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        q = [i]
        for j, y in enumerate(b, 1):
            q.append(min(q[-1] + 1, p[j] + 1, p[j-1] + (x != y)))
        p = q
    return p[-1]


class Evaluator:
    def __init__(self, anchor: str | Path):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        torch.set_num_threads(4)
        self.torch = torch
        self.asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
        self.speaker = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/mari-prosody-ecapa",
            run_opts={"device": "cpu"},
        )
        self.anchor = self.embedding(anchor)

    def load16(self, path: str | Path) -> np.ndarray:
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        y = y.mean(1)
        g = gcd(sr, 16000)
        return resample_poly(y, 16000 // g, sr // g).astype("float32")

    def embedding(self, path: str | Path) -> np.ndarray:
        y = self.load16(path)
        with self.torch.no_grad():
            e = self.speaker.encode_batch(self.torch.from_numpy(y).unsqueeze(0)).squeeze().cpu().numpy()
        return e / max(1e-12, np.linalg.norm(e))

    def evaluate(self, path: str | Path, text: str) -> Dict[str, Any]:
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != 24000 or y.shape[1] != 1:
            raise ValueError("qualification output must be 24 kHz mono")
        segs, _ = self.asr.transcribe(
            str(path), language="en", beam_size=5, temperature=0,
            condition_on_previous_text=False, vad_filter=False,
        )
        hyp = " ".join(s.text.strip() for s in segs)
        expected = normalized(text)
        wer = edit_distance(expected, normalized(hyp)) / max(1, len(expected))
        sim = float(np.dot(self.anchor, self.embedding(path)))
        return {
            "sha256": sha256(path),
            "duration_s": len(y) / sr,
            "asr_text": hyp,
            "wer": wer,
            "speaker_similarity": sim,
            "intelligibility_pass": wer <= 0.12,
            "identity_pass": sim >= 0.60,
        }


def build_basis(root: Path, out_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    finality = load_direction()[0].astype(np.float32)
    attack = np.load(root / "continuation" / "attack_calibration" / "attack_bank.npz")["directions"][0].astype(np.float32)
    tempo = np.load(root / "continuation" / "prosody_basis_tempo" / "tempo_raw.npz")["direction"].astype(np.float32)
    focus = np.load(root / "continuation" / "prosody_basis_focus" / "focus_raw.npz")["direction"].astype(np.float32)

    processed = [finality]
    scales = {"finality_continuation": 1.0}
    correlations = {}

    for name, raw in [("onset_pressure", attack), ("tempo", tempo), ("local_focus", focus)]:
        v = orthogonalize(raw, processed)
        v, scale = scale_like(v, finality)
        for old_name, old in zip(AXES[:len(processed)], processed):
            denom = max(1e-12, float(np.linalg.norm(v) * np.linalg.norm(old)))
            correlations[f"{name}_vs_{old_name}"] = float(np.sum(v * old) / denom)
        processed.append(v)
        scales[name] = scale

    bank = np.stack(processed).astype(np.float32)
    if bank.shape != (4, 29, 2048) or not np.isfinite(bank).all():
        raise ValueError("invalid candidate basis shape")
    np.savez(out_dir / "candidate_basis.npz", directions=bank, axis_names=np.asarray(AXES))
    focus_cal = json.loads((root / "continuation" / "prosody_basis_focus" / "CALIBRATION.json").read_text())
    manifest = {
        "schema": "mari-native-prosody-candidate-basis/1.0",
        "axes": AXES,
        "axis_scales_to_finality_norm": scales,
        "post_orthogonalization_correlations": correlations,
        "focus_lag_frames": float(focus_cal["focus_lag_frames_median"]),
        "basis_npz_sha256": sha256(out_dir / "candidate_basis.npz"),
        "calibration_sources": {
            "finality": "pre-existing measured same-speaker finality/continuation direction",
            "onset": sha256(root / "continuation" / "attack_calibration" / "attack_bank.npz"),
            "tempo": sha256(root / "continuation" / "prosody_basis_tempo" / "tempo_raw.npz"),
            "focus": sha256(root / "continuation" / "prosody_basis_focus" / "focus_raw.npz"),
        },
    }
    write_json(out_dir / "CANDIDATE_BASIS.json", manifest)
    return bank, manifest


def release_for_text(tokenizer: Any, text: str, path: Path) -> Tuple[List[int], List[Tuple[int,int]]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = [tuple(x) for x in enc["offset_mapping"]]
    frames = [0] + list(range(len(ids)))
    write_release(path, ids, frames)
    return ids, offsets


def render_axis(
    root: Path,
    bank: np.ndarray,
    tokenizer: Any,
    text: str,
    output: Path,
    *,
    seed: int,
    axis: int | None = None,
    strength: float = 0.0,
    focus_target: str | None = None,
    focus_lag: float = 0.0,
) -> Dict[str, Any]:
    packet_dir = output.parent / (output.stem + "_packets")
    packet_dir.mkdir(parents=True, exist_ok=True)
    release_path = packet_dir / "release.mrl"
    ids, offsets = release_for_text(tokenizer, text, release_path)
    frame_count = 320
    weights = np.zeros((frame_count, len(AXES)), dtype=np.float32)
    onset = np.zeros((len(AXES),), dtype=np.float32)
    focus_meta = None

    if axis is not None:
        if axis == 1:
            onset[axis] = strength
            for f in range(5):
                weights[f, axis] = strength * max(0.0, 1.0 - f / 5.0)
        elif axis == 2:
            weights[:, axis] = strength
        elif axis == 3:
            if not focus_target:
                raise ValueError("focus target required")
            fake_alignment = {"words": []}
            for m in re.finditer(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", text):
                fake_alignment["words"].append({"word": normalized(m.group(0))[0], "start": 0.0, "end": 0.1})
            target_index, _ = locate_target(fake_alignment, focus_target)
            tok_start, tok_end = target_token_span(tokenizer, text, fake_alignment, target_index)
            release_frame = max(0, tok_end - 1)
            center = int(round(release_frame + focus_lag))
            first = max(0, center - 4)
            last = min(frame_count, center + 5)
            for f in range(first, last):
                u = (f - first) / max(1, last - first - 1)
                weights[f, axis] = strength * math.sin(math.pi * u) ** 2
            focus_meta = {
                "target": focus_target, "target_index": target_index,
                "target_token_span": [tok_start, tok_end],
                "release_frame": release_frame, "predicted_acoustic_center_frame": center,
                "native_window": [first, last],
            }
        else:
            raise ValueError("axis qualification only supports new axes")

    traj = packet_dir / "basis.mtraj"
    t = write_trajectory(traj, bank, weights, onset=onset)
    rec = render(
        root, text, output, seed=seed, trajectory=traj,
        incremental_text=True, text_release=release_path, profile_mode="xvector",
    )
    return {
        "native_record": rec,
        "trajectory": t,
        "focus_projection": focus_meta,
        "strength": strength,
        "axis": None if axis is None else AXES[axis],
    }


def qualify(root: str | Path) -> Dict[str, Any]:
    root = Path(root)
    out = root / "continuation" / "native_prosody_basis_v1"
    out.mkdir(parents=True, exist_ok=True)
    bank, manifest = build_basis(root, out)
    tokenizer = AutoTokenizer.from_pretrained(root / "models" / "qwen-custom", local_files_only=True)
    anchor = root / "recovered" / "production_release" / "production" / "MARI_VOICE_V1_ANCHOR.wav"
    ev = Evaluator(anchor)

    report: Dict[str, Any] = {
        "schema": "mari-native-prosody-basis-qualification/1.0",
        "basis_manifest": manifest,
        "axes": {
            "finality_continuation": {
                "status": "PREEXISTING_QUALIFIED",
                "source": "MARI native finality/continuation calibration lineage",
            }
        },
        "perceptual_naturalness_proven": False,
    }

    # Onset pressure
    onset_rows = []
    for i, text in enumerate([
        "All right. I will check the lock.",
        "I know you have had a difficult day.",
    ]):
        row = {"text": text, "seed": 171100+i, "conditions": {}}
        for label, strength in [("soft", -0.30), ("base", 0.0), ("firm", 0.30)]:
            p = out / f"onset_{i}_{label}.wav"
            render_axis(root, bank, tokenizer, text, p, seed=171100+i,
                        axis=None if label=="base" else 1, strength=strength)
            q = ev.evaluate(p, text)
            q["attack"] = attack_measure(p)
            row["conditions"][label] = q
        soft, base, firm = (row["conditions"][x] for x in ("soft","base","firm"))
        row["effect_db"] = firm["attack"]["initial_to_body_db"] - soft["attack"]["initial_to_body_db"]
        row["pass"] = bool(
            row["effect_db"] >= 0.35 and
            all(x["intelligibility_pass"] and x["identity_pass"] for x in (soft,base,firm)) and
            soft["speaker_similarity"] >= base["speaker_similarity"] - 0.12 and
            firm["speaker_similarity"] >= base["speaker_similarity"] - 0.12
        )
        onset_rows.append(row)
    onset_pass = all(x["pass"] for x in onset_rows)
    report["axes"]["onset_pressure"] = {
        "status": "QUALIFIED" if onset_pass else "REJECTED",
        "orientation": "positive = firmer acoustic onset",
        "rows": onset_rows,
        "gate": "held-out firm-soft onset contrast >=0.35 dB, WER<=.12, speaker similarity>=.60 and within .12 of baseline",
    }

    # Tempo
    tempo_rows = []
    for i, text in enumerate([
        "I checked the front room and the kitchen. Nothing is missing.",
        "Give me the sequence again and I will trace it from the beginning.",
    ]):
        row = {"text": text, "seed": 171200+i, "conditions": {}}
        for label, strength in [("slow", -0.24), ("base", 0.0), ("fast", 0.24)]:
            p = out / f"tempo_{i}_{label}.wav"
            render_axis(root, bank, tokenizer, text, p, seed=171200+i,
                        axis=None if label=="base" else 2, strength=strength)
            row["conditions"][label] = ev.evaluate(p, text)
        slow, base, fast = (row["conditions"][x] for x in ("slow","base","fast"))
        row["slow_to_fast_duration_ratio"] = slow["duration_s"] / max(1e-6, fast["duration_s"])
        row["pass"] = bool(
            row["slow_to_fast_duration_ratio"] >= 1.04 and
            all(x["intelligibility_pass"] and x["identity_pass"] for x in (slow,base,fast)) and
            slow["speaker_similarity"] >= base["speaker_similarity"] - 0.12 and
            fast["speaker_similarity"] >= base["speaker_similarity"] - 0.12
        )
        tempo_rows.append(row)
    tempo_pass = all(x["pass"] for x in tempo_rows)
    report["axes"]["tempo"] = {
        "status": "QUALIFIED" if tempo_pass else "REJECTED",
        "orientation": "positive = faster",
        "rows": tempo_rows,
        "gate": "held-out slow/fast duration ratio>=1.04, WER<=.12, speaker identity gate",
    }

    # Local focus using the online token-release + learned-lag projection.
    focus_rows = []
    for i, (text, target) in enumerate([
        ("Put the red folder on the table.", "red"),
        ("Check the upper drawer before you leave.", "upper"),
    ]):
        row = {"text": text, "target": target, "seed": 171300+i, "conditions": {}}
        for label, strength in [("suppressed", -0.24), ("base", 0.0), ("focused", 0.24)]:
            p = out / f"focus_{i}_{label}.wav"
            native = render_axis(
                root, bank, tokenizer, text, p, seed=171300+i,
                axis=None if label=="base" else 3, strength=strength,
                focus_target=target, focus_lag=manifest["focus_lag_frames"],
            )
            q = ev.evaluate(p, text)
            alignment = exact_word_alignment(ev.asr, p, text)
            target_index, _ = locate_target(alignment, target)
            q["focus"] = focus_score(p, alignment, target_index)
            q["projection"] = native["focus_projection"]
            row["conditions"][label] = q
        sup, base, foc = (row["conditions"][x] for x in ("suppressed","base","focused"))
        row["focused_minus_suppressed_score"] = foc["focus"]["coupled_focus_score"] - sup["focus"]["coupled_focus_score"]
        row["focused_minus_base_score"] = foc["focus"]["coupled_focus_score"] - base["focus"]["coupled_focus_score"]
        row["pass"] = bool(
            row["focused_minus_suppressed_score"] >= 0.45 and
            row["focused_minus_base_score"] >= 0.15 and
            all(x["intelligibility_pass"] and x["identity_pass"] for x in (sup,base,foc)) and
            sup["speaker_similarity"] >= base["speaker_similarity"] - 0.12 and
            foc["speaker_similarity"] >= base["speaker_similarity"] - 0.12
        )
        focus_rows.append(row)
    focus_pass = all(x["pass"] for x in focus_rows)
    report["axes"]["local_focus"] = {
        "status": "QUALIFIED" if focus_pass else "REJECTED",
        "orientation": "positive = stronger local coupled prominence",
        "rows": focus_rows,
        "projection": "online target token release + calibration-median acoustic lag",
        "gate": "held-out focus score ordering, WER<=.12, speaker identity gate",
    }

    qualified = ["finality_continuation"] + [
        name for name, passed in [
            ("onset_pressure", onset_pass),
            ("tempo", tempo_pass),
            ("local_focus", focus_pass),
        ] if passed
    ]
    report["qualified_axes"] = qualified
    report["new_qualified_axis_count"] = len(qualified) - 1
    report["status"] = "PASS" if report["new_qualified_axis_count"] >= 1 else "NO_NEW_AXIS_QUALIFIED"
    report["proof_ceiling"] = (
        "Qualification establishes held-out free-generation acoustic consequence, intelligibility, "
        "and speaker-similarity gates for admitted axes. It does not establish conversational naturalness "
        "or that combining multiple axes is perceptually superior."
    )
    write_json(out / "QUALIFICATION.json", report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    args = ap.parse_args()
    report = qualify(args.root)
    print(json.dumps(report, indent=2), flush=True)
    if report["new_qualified_axis_count"] < 1:
        raise SystemExit("no new native prosody axis qualified")


if __name__ == "__main__":
    main()
