"""Reproducible coach traces; this does not execute or evaluate speech synthesis."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from authentic_performance import ActingContext, coach
from scene_contract import ActionBeat, SceneValidationError

CORPUS = Path(__file__).with_name("corpus.json")


def case_input(case):
    if case["field"] == "text":
        return case["value"], ActingContext()
    if case["field"] == "beat":
        return "Wait. I understand. Let's try again.", ActingContext(
            beats=[ActionBeat(case["when"], case["value"])])
    return "Wait. I understand. Let's try again.", ActingContext(**{case["field"]: case["value"]})


def probe(case):
    text, context = case_input(case)
    try:
        brief = coach(text, context)
    except SceneValidationError as exc:
        findings = exc.to_dict()["findings"]
        outcome = "rejected" if any(f["status"] == "rejected" for f in findings) else "unresolved"
        return {"outcome": outcome, "findings": findings,
                "compilation": "blocked", "renderer_admission": "blocked_before_invocation"}
    return {
        "outcome": "accepted", "findings": [],
        "interpretations": brief.intent_interpretations,
        "compilation": {
            "instruction_sha256": hashlib.sha256(brief.renderer_instruction.encode()).hexdigest(),
            "original_in_instruction": json.dumps(case["value"], ensure_ascii=False) in brief.renderer_instruction,
            "dialogue_unchanged": brief.text == text,
        },
        "renderer_admission": "permitted_by_coach_not_executed",
    }


def meets_expectation(case, result):
    return (result["outcome"] != "accepted" if case["expect"] == "blocked"
            else result["outcome"] == case["expect"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    args = parser.parse_args()
    cases = json.loads(args.corpus.read_text())["cases"]
    records = [{"id": case["id"], "class": case["class"], "expect": case["expect"],
                "trace": probe(case)} for case in cases]
    counts = Counter((r["class"], r["trace"]["outcome"]) for r in records)
    report = {
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scene_contract_sha256": hashlib.sha256((ROOT / "scene_contract.py").read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "evidence": "coach execution only; renderer transport separately tested with executable fixture",
        "counts": {f"{kind}/{outcome}": count for (kind, outcome), count in sorted(counts.items())},
        "cases": records,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["counts"], indent=2))
    mismatches = []
    for case, record in zip(cases, records):
        if not meets_expectation(case, record["trace"]):
            mismatches.append(f'{case["id"]}: expected {case["expect"]}, got {record["trace"]["outcome"]}')
    print("\n".join(mismatches) if mismatches else "All predeclared admission expectations met.")
    return bool(mismatches)


if __name__ == "__main__":
    raise SystemExit(main())
