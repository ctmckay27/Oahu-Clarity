from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class DimensionRole(str, Enum):
    IDENTITY_SEARCH = "identity_search"
    ELASTIC_IDENTITY = "elastic_identity"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    HARD_CONSTRAINT = "hard_constraint"


class EvidenceClass(str, Enum):
    STRUCTURAL_SOURCE = "structural_source"
    HISTORICAL_PRIOR = "historical_prior"
    GENERATED_CANDIDATE = "generated_candidate"
    CARL_LISTENING = "carl_listening"
    OBJECTIVE_MEASUREMENT = "objective_measurement"
    RENDERER_RECEIPT = "renderer_receipt"


class ExperimentPurpose(str, Enum):
    RENDERER_CALIBRATION = "renderer_calibration"
    PROMOTABLE_CANDIDATE = "promotable_candidate"
    IDENTITY_HARDENING = "identity_hardening"
    PERFORMANCE_TEST = "performance_test"
    LONG_FORM_TEST = "long_form_test"


class CandidateStatus(str, Enum):
    GENERATED = "generated"
    CALIBRATION_ONLY = "calibration_only"
    QUALITY_REJECTED = "quality_rejected"
    QUALITY_PASSED_IDENTITY_OPEN = "quality_passed_identity_open"
    IDENTITY_REJECTED = "identity_rejected"
    PLAUSIBLE_MARI = "plausible_mari"
    PROMOTION_BLOCKED = "promotion_blocked"
    ACCEPTED_ANCHOR = "accepted_anchor"
    SUPERSEDED = "superseded"


class LayerPhase(str, Enum):
    CALIBRATING = "calibrating"
    SEARCHING = "searching"
    ANCHOR_CANDIDATE = "anchor_candidate"
    ANCHOR_ACCEPTED = "anchor_accepted"
    HARDENING = "hardening"
    V1_READY = "v1_ready"


class HumanVerdict(str, Enum):
    OPEN = "open"
    REJECT = "reject"
    PASS = "pass"
    PLAUSIBLE = "plausible"
    ACCEPT = "accept"


@dataclass
class CoordinateBelief:
    """Renderer-search coordinate, not a physical fact about Mari."""
    name: str
    role: DimensionRole
    negative_label: str
    positive_label: str
    center: float = 0.0
    radius: float = 1.0
    confidence: float = 0.0
    evidence_count: int = 0
    source_status: str = "OPEN"
    notes: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not -1.0 <= self.center <= 1.0:
            raise ValueError(f"{self.name}.center out of range")
        if not 0.0 <= self.radius <= 1.0:
            raise ValueError(f"{self.name}.radius out of range")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"{self.name}.confidence out of range")


@dataclass
class QualityJudgment:
    naturalness: float
    whole_utterance_coherence: float
    prosody: float
    articulation: float
    human_texture: float
    artifact_cleanliness: float
    aesthetic_quality: float
    long_form_stability: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def values(self, include_long_form: bool = False) -> List[float]:
        vals = [self.naturalness, self.whole_utterance_coherence, self.prosody,
                self.articulation, self.human_texture, self.artifact_cleanliness,
                self.aesthetic_quality]
        if include_long_form and self.long_form_stability is not None:
            vals.append(self.long_form_stability)
        return vals

    def validate(self) -> None:
        for v in self.values(include_long_form=True):
            if not 0.0 <= v <= 1.0:
                raise ValueError("quality values must be in [0, 1]")


@dataclass
class IdentityJudgment:
    same_subject: float
    mari_identity: float
    adult_feminine: float
    agency_presence: float
    non_genericity: float
    accepted_as_mari: bool = False
    notes: List[str] = field(default_factory=list)

    def validate(self) -> None:
        for v in [self.same_subject, self.mari_identity, self.adult_feminine,
                  self.agency_presence, self.non_genericity]:
            if not 0.0 <= v <= 1.0:
                raise ValueError("identity values must be in [0, 1]")


@dataclass
class PerformanceState:
    label: str = "neutral"
    emotion: str = "neutral"
    intensity: float = 0.0
    pace: float = 0.0
    projection: float = 0.0
    intimacy: float = 0.0
    certainty: float = 0.0
    emphasis: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class RendererReceipt:
    renderer_id: str
    implementation_id: str
    model_id: str
    model_revision: str
    seed: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)
    output_sha256: Optional[str] = None
    reference_sha256: Optional[str] = None
    profile_sha256: Optional[str] = None
    network_required_at_inference: Optional[bool] = None
    no_fallback_verified: bool = False


@dataclass
class ExperimentRecord:
    experiment_id: str
    created_at: str
    parent_state_digest: str
    text: str
    performance: PerformanceState
    controls: Dict[str, float]
    prompt: str
    renderer: RendererReceipt
    purpose: ExperimentPurpose = ExperimentPurpose.RENDERER_CALIBRATION
    candidate_status: CandidateStatus = CandidateStatus.GENERATED
    quality: Optional[QualityJudgment] = None
    identity: Optional[IdentityJudgment] = None
    carl_observations: List[str] = field(default_factory=list)
    retained_dimensions: List[str] = field(default_factory=list)
    desired_movements: Dict[str, float] = field(default_factory=dict)
    avoid_traits: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ListeningFeedback:
    candidate_id: str
    recorded_at: str
    actor: str = "Carl"
    quality_verdict: HumanVerdict = HumanVerdict.OPEN
    identity_verdict: HumanVerdict = HumanVerdict.OPEN
    observations: List[str] = field(default_factory=list)
    desired_movements: Dict[str, float] = field(default_factory=dict)
    retain_dimensions: List[str] = field(default_factory=list)
    avoid_traits: List[str] = field(default_factory=list)


@dataclass
class AnchorRecord:
    anchor_id: str
    source_experiment_id: str
    accepted_by: str
    accepted_at: str
    waveform_sha256: str
    renderer_id: str
    profile_sha256: Optional[str] = None
    status: str = "ACCEPTED_NEUTRAL_ACOUSTIC_ANCHOR"
    scope: str = "ordinary neutral speaking voice"
    notes: List[str] = field(default_factory=list)


@dataclass
class RendererCalibration:
    renderer_id: str
    implementation_id: str
    model_id: str
    model_revision: str
    role: str
    axis_language: Dict[str, Dict[str, str]] = field(default_factory=dict)
    hard_prompt_constraints: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class SearchPlan:
    plan_id: str
    generated_at: str
    parent_state_digest: str
    purpose: ExperimentPurpose
    target_dimensions: List[str]
    fixed_dimensions: List[str]
    candidate_controls: List[Dict[str, float]]
    rationale: List[str]
    text: str
    performance: PerformanceState
    renderer_id: str


@dataclass
class ReadinessGateResult:
    gate: str
    passed: bool
    evidence: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


@dataclass
class ReadinessReport:
    phase: LayerPhase
    ready: bool
    gates: List[ReadinessGateResult]


def to_primitive(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_primitive(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_primitive(v) for v in obj]
    return obj


def dataclass_dict(obj: Any) -> Dict[str, Any]:
    return to_primitive(obj)