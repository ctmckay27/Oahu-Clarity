from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .models import IdentityJudgment, QualityJudgment


@dataclass
class QualityAdmissionPolicy:
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "naturalness": 0.82,
        "whole_utterance_coherence": 0.86,
        "prosody": 0.76,
        "articulation": 0.76,
        "human_texture": 0.76,
        "artifact_cleanliness": 0.88,
        "aesthetic_quality": 0.80,
    })
    long_form_threshold: float = 0.80

    def evaluate(self, q: QualityJudgment, require_long_form: bool = False) -> Tuple[bool, List[str]]:
        q.validate()
        failures: List[str] = []
        for name, threshold in self.thresholds.items():
            value = getattr(q, name)
            if value < threshold:
                failures.append(f"{name}={value:.3f} < {threshold:.3f}")
        if require_long_form:
            if q.long_form_stability is None:
                failures.append("long_form_stability missing")
            elif q.long_form_stability < self.long_form_threshold:
                failures.append(
                    f"long_form_stability={q.long_form_stability:.3f} < {self.long_form_threshold:.3f}"
                )
        return not failures, failures


@dataclass
class IdentityAdmissionPolicy:
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "same_subject": 0.82,
        "mari_identity": 0.80,
        "adult_feminine": 0.92,
        "agency_presence": 0.80,
        "non_genericity": 0.72,
    })

    def evaluate(self, j: IdentityJudgment) -> Tuple[bool, List[str]]:
        j.validate()
        failures: List[str] = []
        for name, threshold in self.thresholds.items():
            value = getattr(j, name)
            if value < threshold:
                failures.append(f"{name}={value:.3f} < {threshold:.3f}")
        return not failures, failures