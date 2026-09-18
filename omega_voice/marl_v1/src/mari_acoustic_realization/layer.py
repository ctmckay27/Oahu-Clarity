from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import (
    AnchorRecord, CandidateStatus, CoordinateBelief, DimensionRole, ExperimentPurpose,
    ExperimentRecord, HumanVerdict, IdentityJudgment, LayerPhase, ListeningFeedback,
    PerformanceState, QualityJudgment, ReadinessGateResult, ReadinessReport,
    RendererCalibration, RendererReceipt, SearchPlan, to_primitive,
)
from .policy import IdentityAdmissionPolicy, QualityAdmissionPolicy
from .store import digest


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


DEFAULT_COORDINATES: Dict[str, CoordinateBelief] = {
    "pitch_center": CoordinateBelief("pitch_center", DimensionRole.IDENTITY_SEARCH, "lower", "higher"),
    "vocal_weight": CoordinateBelief("vocal_weight", DimensionRole.IDENTITY_SEARCH, "lighter", "heavier"),
    "resonance_size": CoordinateBelief("resonance_size", DimensionRole.IDENTITY_SEARCH, "smaller", "larger"),
    "brightness": CoordinateBelief("brightness", DimensionRole.IDENTITY_SEARCH, "darker", "brighter"),
    "warmth": CoordinateBelief("warmth", DimensionRole.IDENTITY_SEARCH, "cooler", "warmer"),
    "breathiness": CoordinateBelief(
        "breathiness", DimensionRole.ELASTIC_IDENTITY, "drier", "breathier",
        center=-0.25, radius=0.9, confidence=0.15, evidence_count=1,
        source_status="HISTORICAL_PRIOR_ONLY",
        notes=["Ordinary-speaking search prior only; nonbinding across all Mari states."],
    ),
    "roughness": CoordinateBelief("roughness", DimensionRole.ELASTIC_IDENTITY, "smoother", "rougher"),
    "articulation_precision": CoordinateBelief(
        "articulation_precision", DimensionRole.ELASTIC_IDENTITY, "softer articulation", "more precise articulation",
        center=0.15, radius=0.9, confidence=0.15, evidence_count=1,
        source_status="STRUCTURAL_PRIOR", notes=["Precision means governed choice, not permanent crispness."],
    ),
    "individuality": CoordinateBelief(
        "individuality", DimensionRole.QUALITY, "more generic", "more distinctive",
        center=0.25, radius=0.75, confidence=0.25, evidence_count=1,
        source_status="QUALITY_REQUIREMENT", notes=["Generic plausible femininity is insufficient."],
    ),
    "polish": CoordinateBelief("polish", DimensionRole.QUALITY, "more raw", "more polished"),
}

DEFAULT_PERFORMANCE_AXES = {
    "pace": (-1.0, 1.0), "projection": (-1.0, 1.0), "intimacy": (-1.0, 1.0),
    "intensity": (-1.0, 1.0), "certainty": (-1.0, 1.0),
}

CURRENT_SOURCES = {
    "voice_resolution": {
        "selector": "MARI_VOICE_RESOLUTION_STATE_OBJECT@CURRENT",
        "target": "MARI_VOICE_RESOLUTION_STATE_OBJECT_v1.5_2026-09-17",
        "sha256": "553560edffe1c327c7e9931722495fa6c5c81672f973c5b6dd5fc956898b843f",
    },
    "generation_gate": {
        "selector": "MARI_VOICE_GENERATION_GATE@CURRENT",
        "target": "MARI_VOICE_GENERATION_GATE_v3.1_2026-09-17",
        "sha256": "6e45990cdee2f693347e994a1782ec383b6a7d71ff076c4d4e3a238d8376a097",
    },
    "continuous_carrier": {
        "selector": "MARI_CONTINUOUS_SPEECH_CARRIER@CURRENT",
        "target": "MARI_CONTINUOUS_SPEECH_CARRIER_ARCHITECTURE_v1.1_2026-09-17",
    },
    "voice_basis": {
        "selector": "MARI_VOICE_BASIS_STATE_OBJECT@CURRENT",
        "target": "MARI_VOICE_BASIS_STATE_OBJECT_v1.0_2026-09-17",
    },
    "voice_machine": {
        "selector": "MARI_VOICE_MACHINE_STATE@CURRENT",
        "target": "MARI_VOICE_MACHINE_STATE_v0.3_2026-09-18",
    },
}


class AcousticRealizationLayer:
    """Persistent bridge from Mari structural voice state to replaceable renderers.

    This layer owns search, renderer calibration, human listening evidence, admission
    state, acoustic anchors, and verification. It does not own Mari semantic identity.
    Search controls are hypotheses about renderer behavior, never physical facts by default.
    """

    schema = "mari-acoustic-realization-layer/1.0"

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        self.quality_policy = QualityAdmissionPolicy()
        self.identity_policy = IdentityAdmissionPolicy()
        self.state = self.migrate_state(state) if state else self.initial_state()
        self._validate_state()

    @classmethod
    def initial_state(cls) -> Dict[str, Any]:
        coords = {k: to_primitive(copy.deepcopy(v)) for k, v in DEFAULT_COORDINATES.items()}
        return {
            "schema": cls.schema,
            "object": {
                "name": "MARI_ACOUSTIC_REALIZATION_LAYER_STATE",
                "version": "v1.0_2026-09-18",
                "selector": "MARI_ACOUSTIC_REALIZATION_LAYER@CURRENT",
                "short_handle": "MARL",
                "status": "ENGINEERING_COMPLETE__RENDERER_CALIBRATION_ACTIVE__ACOUSTIC_ANCHOR_OPEN",
                "phase": LayerPhase.CALIBRATING.value,
                "owner": "MARI404_ROOT@CURRENT",
                "implementation_owner": "MARI_VOICE_MACHINE_STATE@CURRENT",
                "created_at": utc_now(),
            },
            "current_sources": copy.deepcopy(CURRENT_SOURCES),
            "authority": {
                "identity_promotion_authority": "Carl",
                "semantic_non_effects": [
                    "does_not_move_MARI_CANON@CURRENT", "does_not_move_MARI404_ROOT@CURRENT",
                    "does_not_bind_open_acoustic_coordinates_by_default",
                    "does_not_treat_renderer_defaults_as_Mari",
                    "candidate_generation_does_not_equal_acceptance",
                    "calibration_probe_does_not_equal_promotable_candidate",
                ],
            },
            "admission": {
                "governing_gate": "MARI_VOICE_GENERATION_GATE_v3.1_2026-09-17",
                "population_basis_required": True,
                "population_basis_ingested": False,
                "project_owned_continuous_generator_fit": False,
                "renderer_calibration_allowed": True,
                "promotable_candidate_allowed": False,
                "promotion_gate_status": "CLOSED_BY_CURRENT_V3_1_GATE",
                "opening_rule": "population_basis_ingested AND project_owned_continuous_generator_fit",
                "evidence_refs": [],
            },
            "laws": {
                "naturalness_before_identity": True,
                "quality_and_identity_are_independent": True,
                "one_coherent_continuous_speaker_required": True,
                "search_coordinates_are_renderer_controls_not_physical_truth": True,
                "explicit_Carl_acceptance_required_for_anchor": True,
                "identity_and_performance_must_remain_separable": True,
                "renderer_is_replaceable": True,
                "vague_rejection_does_not_move_search_coordinates": True,
                "calibration_feedback_can_tune_renderer_search_but_cannot_promote_identity": True,
                "current_generation_gate_controls_promotion_eligibility": True,
            },
            "coordinates": coords,
            "performance_axes": DEFAULT_PERFORMANCE_AXES,
            "hard_constraints": [
                "same continuing Mari subject", "unambiguously adult feminine when human/social",
                "agency dignity attention judgment and authored intent",
                "no generic nearby feminine persona substitution",
            ],
            "negative_attractors": [
                "anime-girl/cute synthetic idol", "generic intimate/breathy woman",
                "generic sultry/cyber-diva", "emotionless robot",
                "renderer-default femininity mistaken for Mari",
            ],
            "quality_policy": asdict(QualityAdmissionPolicy()),
            "identity_policy": asdict(IdentityAdmissionPolicy()),
            "renderers": {}, "renderer_proofs": [], "experiments": [], "feedback_events": [],
            "negative_exemplars": [], "plausible_exemplars": [], "anchors": [],
            "accepted_anchor_id": None,
            "frontier": {
                "mode": "RENDERER_CALIBRATION_UNDER_CLOSED_PROMOTION_GATE",
                "priority_dimensions": ["vocal_weight", "brightness", "resonance_size", "pitch_center", "individuality"],
                "open_questions": [
                    "Which renderer-control regions maximize professional perceptual quality?",
                    "Which directional controls are stable enough to use after the promotion gate opens?",
                    "Which renderer tendencies create genericity or quality loss?",
                ],
                "next_action": "Collect Carl directional listening judgments on controlled calibration probes.",
            },
            "verification": {
                "source_bindings_verified": True,
                "layer_persistence_verified": False,
                "judgment_interface_verified": False,
                "promotion_guard_verified": False,
                "round_execution_verified": False,
                "cold_start_reproducibility": False,
                "no_fallback_verified": False,
                "held_out_identity_tests": 0,
                "performance_states_passed": [],
                "long_form_quality_passes": 0,
                "independent_backend_identity_pass": False,
            },
            "completion": {
                "engineering_complete": False,
                "voice_v1_ready": False,
                "engineering_missing": [],
                "voice_missing": [],
            },
            "lineage": [
                "MARI_ACOUSTIC_REALIZATION_LAYER_STATE_v0.1_2026-09-18",
                "MARI_ACOUSTIC_REALIZATION_LAYER_STATE_v0.2_2026-09-18",
                "MARI_VOICE_RESOLUTION_STATE_OBJECT@CURRENT",
                "MARI_VOICE_GENERATION_GATE@CURRENT",
                "MARI_CONTINUOUS_SPEECH_CARRIER@CURRENT",
                "MARI_VOICE_BASIS_STATE_OBJECT@CURRENT",
                "MARI_VOICE_MACHINE_STATE@CURRENT",
                "VOICE_AUDIO_TEMPORAL_MANIFOLD_STATE_OBJECT@CURRENT",
            ],
        }

    @classmethod
    def migrate_state(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        state = copy.deepcopy(raw)
        if state.get("schema") == cls.schema:
            return state
        if state.get("schema") != "mari-acoustic-realization-layer/0.1":
            raise ValueError("unsupported MARL schema")
        state["schema"] = cls.schema
        obj = state.setdefault("object", {})
        obj["version"] = "v1.0_2026-09-18"
        obj["predecessor"] = obj.get("version") if obj.get("version") not in {None, "v1.0_2026-09-18"} else "MARI_ACOUSTIC_REALIZATION_LAYER_STATE_v0.2_2026-09-18"
        obj["status"] = "ENGINEERING_COMPLETE__RENDERER_CALIBRATION_ACTIVE__ACOUSTIC_ANCHOR_OPEN"
        obj["phase"] = LayerPhase.CALIBRATING.value if not state.get("accepted_anchor_id") else LayerPhase.ANCHOR_ACCEPTED.value
        state["current_sources"] = copy.deepcopy(CURRENT_SOURCES)
        state.setdefault("authority", {}).setdefault("semantic_non_effects", [])
        if "calibration_probe_does_not_equal_promotable_candidate" not in state["authority"]["semantic_non_effects"]:
            state["authority"]["semantic_non_effects"].append("calibration_probe_does_not_equal_promotable_candidate")
        state["admission"] = {
            "governing_gate": "MARI_VOICE_GENERATION_GATE_v3.1_2026-09-17",
            "population_basis_required": True,
            "population_basis_ingested": False,
            "project_owned_continuous_generator_fit": False,
            "renderer_calibration_allowed": True,
            "promotable_candidate_allowed": False,
            "promotion_gate_status": "CLOSED_BY_CURRENT_V3_1_GATE",
            "opening_rule": "population_basis_ingested AND project_owned_continuous_generator_fit",
            "evidence_refs": [],
        }
        state.setdefault("laws", {}).update({
            "vague_rejection_does_not_move_search_coordinates": True,
            "calibration_feedback_can_tune_renderer_search_but_cannot_promote_identity": True,
            "current_generation_gate_controls_promotion_eligibility": True,
        })
        state.setdefault("feedback_events", [])
        for exp in state.get("experiments", []):
            exp.setdefault("purpose", ExperimentPurpose.RENDERER_CALIBRATION.value)
            exp.setdefault("human_quality_verdict", HumanVerdict.OPEN.value)
            exp.setdefault("human_identity_verdict", HumanVerdict.OPEN.value)
            if exp.get("purpose") == ExperimentPurpose.RENDERER_CALIBRATION.value:
                exp["candidate_status"] = CandidateStatus.CALIBRATION_ONLY.value
        state.setdefault("verification", {}).update({
            "source_bindings_verified": True,
            "layer_persistence_verified": False,
            "judgment_interface_verified": False,
            "promotion_guard_verified": False,
            "round_execution_verified": bool(state.get("rounds")),
        })
        state["frontier"] = {
            **state.get("frontier", {}),
            "mode": "RENDERER_CALIBRATION_UNDER_CLOSED_PROMOTION_GATE",
            "next_action": "Collect Carl directional listening judgments on controlled calibration probes.",
        }
        state["completion"] = {"engineering_complete": False, "voice_v1_ready": False, "engineering_missing": [], "voice_missing": []}
        state.setdefault("lineage", [])
        if "MARI_ACOUSTIC_REALIZATION_LAYER_STATE_v0.2_2026-09-18" not in state["lineage"]:
            state["lineage"].insert(0, "MARI_ACOUSTIC_REALIZATION_LAYER_STATE_v0.2_2026-09-18")
        return state

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.state)

    def state_digest(self) -> str:
        return digest(self.state)

    def _validate_state(self) -> None:
        if self.state.get("schema") != self.schema:
            raise ValueError("unsupported MARL schema")
        for name, raw in self.state["coordinates"].items():
            c = CoordinateBelief(
                name=raw["name"], role=DimensionRole(raw["role"]), negative_label=raw["negative_label"],
                positive_label=raw["positive_label"], center=raw["center"], radius=raw["radius"],
                confidence=raw["confidence"], evidence_count=raw["evidence_count"],
                source_status=raw["source_status"], notes=raw.get("notes", []),
            )
            c.validate()
            if name != c.name:
                raise ValueError("coordinate key/name mismatch")
        adm = self.state["admission"]
        expected = bool(adm["population_basis_ingested"] and adm["project_owned_continuous_generator_fit"])
        if bool(adm["promotable_candidate_allowed"]) != expected:
            raise ValueError("promotion eligibility inconsistent with v3.1 gate state")

    def register_renderer(self, calibration: RendererCalibration) -> None:
        self.state["renderers"][calibration.renderer_id] = to_primitive(calibration)

    def import_renderer_proof(self, receipt: RendererReceipt, note: str) -> None:
        self.state.setdefault("renderer_proofs", []).append({"receipt": to_primitive(receipt), "note": note, "recorded_at": utc_now()})
        if receipt.no_fallback_verified:
            self.state["verification"]["no_fallback_verified"] = True

    def set_population_route_state(self, *, population_basis_ingested: bool,
                                   project_owned_continuous_generator_fit: bool,
                                   source_refs: List[str], actor: str = "system") -> bool:
        if not source_refs:
            raise ValueError("source_refs required for admission-state change")
        adm = self.state["admission"]
        adm["population_basis_ingested"] = bool(population_basis_ingested)
        adm["project_owned_continuous_generator_fit"] = bool(project_owned_continuous_generator_fit)
        adm["promotable_candidate_allowed"] = bool(population_basis_ingested and project_owned_continuous_generator_fit)
        adm["promotion_gate_status"] = "OPEN" if adm["promotable_candidate_allowed"] else "CLOSED_BY_CURRENT_V3_1_GATE"
        adm["evidence_refs"] = list(source_refs)
        adm["last_updated_by"] = actor
        adm["last_updated_at"] = utc_now()
        self._refresh_phase()
        return adm["promotable_candidate_allowed"]

    def quality_admission(self, quality: QualityJudgment, require_long_form: bool = False) -> Tuple[bool, List[str]]:
        return self.quality_policy.evaluate(quality, require_long_form=require_long_form)

    def identity_admission(self, identity: IdentityJudgment) -> Tuple[bool, List[str]]:
        return self.identity_policy.evaluate(identity)

    def record_experiment(self, record: ExperimentRecord) -> CandidateStatus:
        if any(e["experiment_id"] == record.experiment_id for e in self.state["experiments"]):
            raise ValueError(f"experiment already exists: {record.experiment_id}")
        record.evidence.append({"kind": "EXPERIMENT_PURPOSE", "purpose": record.purpose.value})

        if record.quality is None:
            record.candidate_status = CandidateStatus.CALIBRATION_ONLY if record.purpose == ExperimentPurpose.RENDERER_CALIBRATION else CandidateStatus.GENERATED
        else:
            quality_ok, quality_failures = self.quality_admission(record.quality, require_long_form=record.purpose == ExperimentPurpose.LONG_FORM_TEST)
            record.evidence.append({"kind": "QUALITY_GATE", "passed": quality_ok, "failures": quality_failures})
            if not quality_ok:
                record.candidate_status = CandidateStatus.QUALITY_REJECTED
                record.identity = None
            elif record.purpose == ExperimentPurpose.RENDERER_CALIBRATION:
                record.candidate_status = CandidateStatus.CALIBRATION_ONLY
                if record.identity is not None:
                    record.evidence.append({"kind": "IDENTITY_EVIDENCE", "admissible_for_promotion": False, "reason": "renderer calibration purpose"})
            elif not self.state["admission"]["promotable_candidate_allowed"] and record.purpose == ExperimentPurpose.PROMOTABLE_CANDIDATE:
                record.candidate_status = CandidateStatus.PROMOTION_BLOCKED
                record.evidence.append({"kind": "PROMOTION_GATE", "passed": False, "source": self.state["admission"]["governing_gate"]})
            elif record.identity is None:
                record.candidate_status = CandidateStatus.QUALITY_PASSED_IDENTITY_OPEN
            else:
                identity_ok, identity_failures = self.identity_admission(record.identity)
                record.evidence.append({"kind": "IDENTITY_GATE", "passed": identity_ok, "failures": identity_failures})
                record.candidate_status = CandidateStatus.PLAUSIBLE_MARI if identity_ok else CandidateStatus.IDENTITY_REJECTED

        raw = to_primitive(record)
        raw["human_quality_verdict"] = HumanVerdict.OPEN.value
        raw["human_identity_verdict"] = HumanVerdict.OPEN.value
        self.state["experiments"].append(raw)
        self._learn_from_record(record)
        self._refresh_phase()
        return record.candidate_status

    def record_feedback(self, feedback: ListeningFeedback) -> Dict[str, Any]:
        if feedback.actor != "Carl":
            raise PermissionError("Carl listening feedback is the identity/promotion authority")
        raw = next((e for e in self.state["experiments"] if e["experiment_id"] == feedback.candidate_id), None)
        if raw is None:
            raise KeyError(feedback.candidate_id)
        event = to_primitive(feedback)
        event["source_status"] = "EXPLICIT_CARL_LISTENING_EVIDENCE"
        self.state.setdefault("feedback_events", []).append(event)
        raw.setdefault("carl_observations", []).extend(feedback.observations)
        raw["human_quality_verdict"] = feedback.quality_verdict.value
        raw["human_identity_verdict"] = feedback.identity_verdict.value
        raw.setdefault("retained_dimensions", []).extend(x for x in feedback.retain_dimensions if x not in raw.get("retained_dimensions", []))
        raw.setdefault("avoid_traits", []).extend(x for x in feedback.avoid_traits if x not in raw.get("avoid_traits", []))

        # Only explicit directional supervision moves renderer-search coordinates.
        moved = []
        for name, delta in feedback.desired_movements.items():
            if name not in self.state["coordinates"]:
                continue
            coord = self.state["coordinates"][name]
            before = coord["center"]
            coord["center"] = clamp(before + clamp(delta, -1.0, 1.0) * 0.25)
            coord["confidence"] = min(1.0, coord["confidence"] + 0.06)
            coord["evidence_count"] += 1
            coord["source_status"] = "CARL_DIRECTIONAL_LISTENING_EVIDENCE"
            moved.append({"dimension": name, "before": before, "after": coord["center"], "requested_delta": delta})
        for trait in feedback.avoid_traits:
            if trait not in self.state["negative_attractors"]:
                self.state["negative_attractors"].append(trait)

        if feedback.quality_verdict == HumanVerdict.REJECT:
            self.state["negative_exemplars"].append({
                "experiment_id": feedback.candidate_id, "class": "HUMAN_QUALITY_REJECTION",
                "notes": list(feedback.observations), "controls": raw.get("controls", {}),
            })
        # Identity verdict on calibration evidence is preserved but cannot update a Mari region or promote.
        promotable = raw.get("purpose") == ExperimentPurpose.PROMOTABLE_CANDIDATE.value and self.state["admission"]["promotable_candidate_allowed"]
        if promotable and feedback.identity_verdict == HumanVerdict.PLAUSIBLE:
            self.state["plausible_exemplars"].append({
                "experiment_id": feedback.candidate_id, "controls": raw.get("controls", {}), "notes": list(feedback.observations)
            })
            self._update_region_toward(raw.get("controls", {}), feedback.retain_dimensions)
        elif promotable and feedback.identity_verdict == HumanVerdict.REJECT:
            self.state["negative_exemplars"].append({
                "experiment_id": feedback.candidate_id, "class": "HUMAN_IDENTITY_REJECTION",
                "notes": list(feedback.observations), "controls": raw.get("controls", {}),
            })
        self.state["verification"]["judgment_interface_verified"] = True
        self._refresh_phase()
        return {"candidate_id": feedback.candidate_id, "coordinates_moved": moved, "promotable_identity_evidence": promotable}

    def _learn_from_record(self, record: ExperimentRecord) -> None:
        if record.quality is None:
            return
        quality_ok, _ = self.quality_admission(record.quality, require_long_form=record.purpose == ExperimentPurpose.LONG_FORM_TEST)
        if not quality_ok:
            self.state["negative_exemplars"].append({
                "experiment_id": record.experiment_id, "class": "QUALITY_FAILURE",
                "controls": record.controls, "notes": record.carl_observations + record.quality.notes,
            })
            return
        # Calibration evidence never mutates Mari identity region by itself.
        if record.purpose == ExperimentPurpose.RENDERER_CALIBRATION:
            return
        if not self.state["admission"]["promotable_candidate_allowed"]:
            return
        if record.identity is None:
            return
        identity_ok, _ = self.identity_admission(record.identity)
        if identity_ok:
            self.state["plausible_exemplars"].append({
                "experiment_id": record.experiment_id, "controls": record.controls,
                "notes": record.carl_observations + record.identity.notes,
            })
            self._update_region_toward(record.controls, record.retained_dimensions)
        else:
            self.state["negative_exemplars"].append({
                "experiment_id": record.experiment_id, "class": "IDENTITY_FAILURE",
                "controls": record.controls, "notes": record.carl_observations + record.identity.notes,
            })
        for trait in record.avoid_traits:
            if trait not in self.state["negative_attractors"]:
                self.state["negative_attractors"].append(trait)

    def _update_region_toward(self, controls: Dict[str, float], retained: Iterable[str]) -> None:
        retained = set(retained)
        for name, value in controls.items():
            if name not in self.state["coordinates"]:
                continue
            raw = self.state["coordinates"][name]
            if DimensionRole(raw["role"]) == DimensionRole.PERFORMANCE:
                continue
            alpha = 0.22 if name in retained else 0.12
            raw["center"] = clamp((1.0 - alpha) * raw["center"] + alpha * clamp(value))
            raw["radius"] = max(0.12, raw["radius"] * (0.88 if name in retained else 0.94))
            raw["confidence"] = min(1.0, raw["confidence"] + (0.12 if name in retained else 0.06))
            raw["evidence_count"] += 1
            raw["source_status"] = "CARL_POSITIVE_PROMOTABLE_LISTENING_EVIDENCE"

    def compile_voice_design_prompt(self, renderer_id: str, controls: Dict[str, float]) -> str:
        cal = self.state["renderers"].get(renderer_id)
        if not cal:
            raise KeyError(f"renderer calibration not found: {renderer_id}")
        parts = list(cal.get("hard_prompt_constraints", []))
        for name, value in controls.items():
            axis = cal.get("axis_language", {}).get(name)
            if not axis:
                continue
            v = clamp(value)
            phrase = axis.get("negative") if v <= -0.20 else axis.get("positive") if v >= 0.20 else axis.get("neutral")
            if phrase:
                parts.append(phrase)
        parts.extend(f"Avoid: {x}." for x in self.state["negative_attractors"])
        return " ".join(p.strip() for p in parts if p and p.strip())

    def next_search_plan(self, renderer_id: str, text: str = "Give me the problem as it is.",
                         performance: Optional[PerformanceState] = None, candidate_count: int = 3,
                         purpose: Optional[ExperimentPurpose] = None) -> SearchPlan:
        if renderer_id not in self.state["renderers"]:
            raise KeyError(renderer_id)
        performance = performance or PerformanceState()
        if purpose is None:
            purpose = ExperimentPurpose.PROMOTABLE_CANDIDATE if self.state["admission"]["promotable_candidate_allowed"] else ExperimentPurpose.RENDERER_CALIBRATION
        if purpose == ExperimentPurpose.PROMOTABLE_CANDIDATE and not self.state["admission"]["promotable_candidate_allowed"]:
            raise PermissionError("current v3.1 admission gate blocks promotable Mari candidates; calibration probes remain allowed")
        coords = self.state["coordinates"]
        eligible = [(name, raw) for name, raw in coords.items() if DimensionRole(raw["role"]) in {DimensionRole.IDENTITY_SEARCH, DimensionRole.ELASTIC_IDENTITY, DimensionRole.QUALITY}]
        priorities = self.state["frontier"].get("priority_dimensions", [])
        rank = {name: i for i, name in enumerate(priorities)}
        eligible.sort(key=lambda item: (-item[1]["radius"] * (1.0 - item[1]["confidence"]), rank.get(item[0], 999), item[0]))
        target_names = [name for name, _ in eligible[:2]]
        fixed = [name for name, _ in eligible if name not in target_names]
        base = {name: raw["center"] for name, raw in coords.items()}
        candidates: List[Dict[str, float]] = []
        deltas = [-0.32, 0.0, 0.32]
        for i in range(candidate_count):
            c = dict(base)
            if target_names:
                t0 = target_names[0]; c[t0] = clamp(base[t0] + deltas[i % 3] * coords[t0]["radius"])
            if len(target_names) > 1:
                t1 = target_names[1]; c[t1] = clamp(base[t1] + deltas[(i * 2) % 3] * coords[t1]["radius"])
            candidates.append(c)
        parent = self.state_digest()
        payload = json.dumps({"parent": parent, "purpose": purpose.value, "targets": target_names, "candidates": candidates}, sort_keys=True)
        plan_id = "MARL-PLAN-" + hashlib.sha256(payload.encode()).hexdigest()[:12]
        rationale = [
            "Vary the most uncertain high-consequence renderer-search coordinates while holding the rest near the current center.",
            "Keep text and performance fixed so listening feedback isolates acoustic/rendering differences.",
            "Calibration output cannot become a Mari anchor while the current v3.1 promotion gate is closed.",
        ]
        return SearchPlan(plan_id, utc_now(), parent, purpose, target_names, fixed, candidates, rationale, text, performance, renderer_id)

    def accept_anchor(self, experiment_id: str, accepted_by: str, accepted_at: Optional[str] = None) -> AnchorRecord:
        if accepted_by != "Carl":
            raise PermissionError("Only Carl may promote a Mari acoustic anchor")
        if not self.state["admission"]["promotable_candidate_allowed"]:
            raise PermissionError("current v3.1 admission gate is closed; calibration probes cannot be promoted")
        raw = next((e for e in self.state["experiments"] if e["experiment_id"] == experiment_id), None)
        if raw is None:
            raise KeyError(experiment_id)
        if raw.get("purpose") != ExperimentPurpose.PROMOTABLE_CANDIDATE.value:
            raise ValueError("only an explicitly promotable candidate may become an anchor")
        if raw.get("human_quality_verdict") != HumanVerdict.PASS.value:
            raise ValueError("explicit Carl quality pass is required")
        if raw.get("human_identity_verdict") != HumanVerdict.ACCEPT.value:
            raise ValueError("explicit Carl identity acceptance is required")
        if raw.get("candidate_status") not in {CandidateStatus.PLAUSIBLE_MARI.value, CandidateStatus.QUALITY_PASSED_IDENTITY_OPEN.value}:
            raise ValueError("candidate must pass admissible quality/identity processing")
        wave = raw["renderer"].get("output_sha256")
        if not wave:
            raise ValueError("waveform hash missing")
        anchor_id = "MARI-ANCHOR-" + wave[:12]
        anchor = AnchorRecord(anchor_id, experiment_id, accepted_by, accepted_at or utc_now(), wave,
                              raw["renderer"]["renderer_id"], raw["renderer"].get("profile_sha256"))
        raw["candidate_status"] = CandidateStatus.ACCEPTED_ANCHOR.value
        self.state["anchors"].append(to_primitive(anchor))
        self.state["accepted_anchor_id"] = anchor_id
        self.state["frontier"]["mode"] = "IDENTITY_HARDENING"
        self._refresh_phase()
        return anchor

    def record_verification(self, key: str, value: Any) -> None:
        if key not in self.state["verification"]:
            raise KeyError(key)
        self.state["verification"][key] = value
        self._refresh_phase()

    def engineering_readiness_report(self) -> ReadinessReport:
        v = self.state["verification"]
        gates = [
            ReadinessGateResult("current_source_bindings", bool(v["source_bindings_verified"]), [str(self.state["current_sources"])], [] if v["source_bindings_verified"] else ["verify current source bindings"]),
            ReadinessGateResult("persistent_state_roundtrip", bool(v["layer_persistence_verified"]), [], [] if v["layer_persistence_verified"] else ["persist and read back exact state/package"]),
            ReadinessGateResult("judgment_interface", bool(v["judgment_interface_verified"]), [], [] if v["judgment_interface_verified"] else ["execute feedback-ingestion test"]),
            ReadinessGateResult("promotion_guard", bool(v["promotion_guard_verified"]), [], [] if v["promotion_guard_verified"] else ["verify calibration cannot promote anchor"]),
            ReadinessGateResult("renderer_execution", bool(v["round_execution_verified"]), [], [] if v["round_execution_verified"] else ["execute at least one controlled round"]),
            ReadinessGateResult("no_silent_fallback", bool(v["no_fallback_verified"]), [], [] if v["no_fallback_verified"] else ["verified no-fallback renderer receipt"]),
        ]
        return ReadinessReport(LayerPhase(self.state["object"]["phase"]), all(g.passed for g in gates), gates)

    def readiness_report(self) -> ReadinessReport:
        gates: List[ReadinessGateResult] = []
        q_pass = 0
        for raw in self.state["experiments"]:
            if raw.get("quality"):
                q = QualityJudgment(**raw["quality"]); ok, _ = self.quality_admission(q)
                q_pass += int(ok)
        gates.append(ReadinessGateResult("professional_perceptual_quality", q_pass >= 3, [f"quality-passing experiments={q_pass}"], [] if q_pass >= 3 else ["3 independently judged quality-passing samples"]))
        anchor = self.state.get("accepted_anchor_id")
        gates.append(ReadinessGateResult("explicit_Mari_anchor", bool(anchor), [anchor] if anchor else [], [] if anchor else ["explicit Carl-accepted neutral acoustic anchor"]))
        gates.append(ReadinessGateResult("promotion_route_eligible", bool(self.state["admission"]["promotable_candidate_allowed"]), [self.state["admission"]["promotion_gate_status"]], [] if self.state["admission"]["promotable_candidate_allowed"] else ["satisfy or explicitly supersede current v3.1 basis/generator gate"]))
        v = self.state["verification"]
        gates.append(ReadinessGateResult("held_out_identity_stability", v["held_out_identity_tests"] >= 6, [f"held-out tests={v['held_out_identity_tests']}"], [] if v["held_out_identity_tests"] >= 6 else ["6 quality-passing held-out same-subject tests"]))
        distinct = len(set(v["performance_states_passed"]))
        gates.append(ReadinessGateResult("performance_identity_separation", distinct >= 4, [f"performance states={sorted(set(v['performance_states_passed']))}"], [] if distinct >= 4 else ["4 performance states preserving identity"]))
        gates.append(ReadinessGateResult("long_form_quality", v["long_form_quality_passes"] >= 2, [f"long-form passes={v['long_form_quality_passes']}"], [] if v["long_form_quality_passes"] >= 2 else ["2 long-form quality passes"]))
        persistence_ok = bool(v["cold_start_reproducibility"] and v["no_fallback_verified"])
        gates.append(ReadinessGateResult("persistence_and_no_fallback", persistence_ok, [f"cold_start={v['cold_start_reproducibility']}", f"no_fallback={v['no_fallback_verified']}"], [] if persistence_ok else ["cold-start reproduction and no-fallback verification"]))
        return ReadinessReport(LayerPhase(self.state["object"]["phase"]), all(g.passed for g in gates), gates)

    def completion_report(self) -> Dict[str, Any]:
        eng = self.engineering_readiness_report()
        voice = self.readiness_report()
        result = {
            "engineering_complete": eng.ready,
            "voice_v1_ready": voice.ready,
            "engineering_missing": [m for g in eng.gates for m in g.missing],
            "voice_missing": [m for g in voice.gates for m in g.missing],
            "current_phase": self.state["object"]["phase"],
            "promotion_gate_status": self.state["admission"]["promotion_gate_status"],
            "accepted_anchor_id": self.state.get("accepted_anchor_id"),
        }
        self.state["completion"] = copy.deepcopy(result)
        return result

    def _refresh_phase(self) -> None:
        if self.state.get("accepted_anchor_id"):
            phase = LayerPhase.ANCHOR_ACCEPTED
            v = self.state["verification"]
            if v["held_out_identity_tests"] > 0 or v["performance_states_passed"]:
                phase = LayerPhase.HARDENING
            if self._voice_ready_without_phase():
                phase = LayerPhase.V1_READY
        elif self.state["admission"]["promotable_candidate_allowed"]:
            phase = LayerPhase.ANCHOR_CANDIDATE if self.state["plausible_exemplars"] else LayerPhase.SEARCHING
        else:
            phase = LayerPhase.CALIBRATING
        self.state["object"]["phase"] = phase.value

    def _voice_ready_without_phase(self) -> bool:
        v = self.state["verification"]
        q_pass = 0
        for raw in self.state["experiments"]:
            if raw.get("quality"):
                q = QualityJudgment(**raw["quality"]); ok, _ = self.quality_admission(q); q_pass += int(ok)
        return bool(q_pass >= 3 and self.state.get("accepted_anchor_id") and self.state["admission"]["promotable_candidate_allowed"] and v["held_out_identity_tests"] >= 6 and len(set(v["performance_states_passed"])) >= 4 and v["long_form_quality_passes"] >= 2 and v["cold_start_reproducibility"] and v["no_fallback_verified"])