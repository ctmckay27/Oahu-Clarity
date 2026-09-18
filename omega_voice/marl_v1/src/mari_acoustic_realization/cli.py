from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .bootstrap import build_current_layer, import_current_renderer_proofs, register_current_renderers
from .layer import AcousticRealizationLayer
from .models import (
    ExperimentPurpose, ExperimentRecord, HumanVerdict, IdentityJudgment, ListeningFeedback,
    PerformanceState, QualityJudgment, RendererReceipt,
)
from .store import atomic_write_json, load_json


def load_layer(path: Path) -> AcousticRealizationLayer:
    if path.exists():
        layer = AcousticRealizationLayer(load_json(path))
        register_current_renderers(layer); import_current_renderer_proofs(layer)
        return layer
    return build_current_layer()


def experiment_from_dict(raw: Dict[str, Any]) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=raw["experiment_id"], created_at=raw["created_at"],
        parent_state_digest=raw["parent_state_digest"], text=raw["text"],
        performance=PerformanceState(**raw.get("performance", {})), controls=raw.get("controls", {}),
        prompt=raw.get("prompt", ""), renderer=RendererReceipt(**raw["renderer"]),
        purpose=ExperimentPurpose(raw.get("purpose", ExperimentPurpose.RENDERER_CALIBRATION.value)),
        quality=QualityJudgment(**raw["quality"]) if raw.get("quality") else None,
        identity=IdentityJudgment(**raw["identity"]) if raw.get("identity") else None,
        carl_observations=raw.get("carl_observations", []), retained_dimensions=raw.get("retained_dimensions", []),
        desired_movements=raw.get("desired_movements", {}), avoid_traits=raw.get("avoid_traits", []),
        evidence=raw.get("evidence", []),
    )


def feedback_from_dict(raw: Dict[str, Any]) -> ListeningFeedback:
    return ListeningFeedback(
        candidate_id=raw["candidate_id"], recorded_at=raw.get("recorded_at", ""), actor=raw.get("actor", "Carl"),
        quality_verdict=HumanVerdict(raw.get("quality_verdict", "open")),
        identity_verdict=HumanVerdict(raw.get("identity_verdict", "open")),
        observations=raw.get("observations", []), desired_movements=raw.get("desired_movements", {}),
        retain_dimensions=raw.get("retain_dimensions", []), avoid_traits=raw.get("avoid_traits", []),
    )


def save(path: Path, layer: AcousticRealizationLayer) -> str:
    return atomic_write_json(path, layer.snapshot())


def main(argv=None):
    p = argparse.ArgumentParser(prog="marl")
    p.add_argument("--state", default="MARI_ACOUSTIC_REALIZATION_LAYER_STATE.json")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init"); sub.add_parser("show"); sub.add_parser("completion")

    nx = sub.add_parser("next")
    nx.add_argument("--renderer", default="qwen3-tts-voicedesign-1.7b")
    nx.add_argument("--text", default="Give me the problem as it is.")
    nx.add_argument("--count", type=int, default=3)
    nx.add_argument("--purpose", choices=[x.value for x in ExperimentPurpose], default=None)

    ing = sub.add_parser("ingest")
    ing.add_argument("--record", required=True)

    fb = sub.add_parser("feedback")
    fb.add_argument("--record", required=True)

    acc = sub.add_parser("accept")
    acc.add_argument("--experiment-id", required=True)
    acc.add_argument("--accepted-by", default="Carl")

    gate = sub.add_parser("gate")
    gate.add_argument("--basis-ingested", required=True, choices=["true","false"])
    gate.add_argument("--generator-fit", required=True, choices=["true","false"])
    gate.add_argument("--source-ref", action="append", required=True)

    ver = sub.add_parser("verify")
    ver.add_argument("--key", required=True); ver.add_argument("--value", required=True)
    sub.add_parser("readiness"); sub.add_parser("engineering-readiness")

    args = p.parse_args(argv); path = Path(args.state); layer = load_layer(path)
    if args.cmd == "init":
        sha = save(path, layer); print(json.dumps({"state":str(path),"sha256":sha,"digest":layer.state_digest()}, indent=2))
    elif args.cmd == "show": print(json.dumps(layer.snapshot(), indent=2, sort_keys=True))
    elif args.cmd == "next":
        purpose = ExperimentPurpose(args.purpose) if args.purpose else None
        plan = layer.next_search_plan(args.renderer, args.text, PerformanceState(), args.count, purpose)
        print(json.dumps({"plan":plan.__dict__,"prompts":[layer.compile_voice_design_prompt(args.renderer,c) for c in plan.candidate_controls]}, indent=2, default=lambda o:o.value if hasattr(o,"value") else o.__dict__))
    elif args.cmd == "ingest":
        rec = experiment_from_dict(json.loads(Path(args.record).read_text())); status=layer.record_experiment(rec); sha=save(path,layer)
        print(json.dumps({"experiment_id":rec.experiment_id,"status":status.value,"state_sha256":sha,"state_digest":layer.state_digest()},indent=2))
    elif args.cmd == "feedback":
        rec=feedback_from_dict(json.loads(Path(args.record).read_text())); result=layer.record_feedback(rec); sha=save(path,layer)
        print(json.dumps({"feedback":result,"state_sha256":sha,"state_digest":layer.state_digest()},indent=2))
    elif args.cmd == "accept":
        anchor=layer.accept_anchor(args.experiment_id,args.accepted_by); sha=save(path,layer)
        print(json.dumps({"anchor":anchor.__dict__,"state_sha256":sha,"state_digest":layer.state_digest()},indent=2))
    elif args.cmd == "gate":
        opened=layer.set_population_route_state(population_basis_ingested=args.basis_ingested=="true", project_owned_continuous_generator_fit=args.generator_fit=="true", source_refs=args.source_ref); sha=save(path,layer)
        print(json.dumps({"promotable_candidate_allowed":opened,"state_sha256":sha},indent=2))
    elif args.cmd == "verify":
        value=json.loads(args.value); layer.record_verification(args.key,value); sha=save(path,layer)
        print(json.dumps({"verification":{args.key:value},"state_sha256":sha},indent=2))
    elif args.cmd == "readiness":
        r=layer.readiness_report(); print(json.dumps({"phase":r.phase.value,"ready":r.ready,"gates":[g.__dict__ for g in r.gates]},indent=2))
    elif args.cmd == "engineering-readiness":
        r=layer.engineering_readiness_report(); print(json.dumps({"phase":r.phase.value,"ready":r.ready,"gates":[g.__dict__ for g in r.gates]},indent=2))
    elif args.cmd == "completion":
        print(json.dumps(layer.completion_report(), indent=2))

if __name__ == "__main__": main()