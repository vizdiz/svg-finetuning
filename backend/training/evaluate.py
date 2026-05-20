"""
training.evaluate

Stub evaluation entrypoint for a SageMaker Processing step.

This writes a deterministic evaluation artifact instead of trying to
perform a heavyweight benchmark before the inference stack is ready.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EvaluationReport:
    model_artifacts: str
    val_records: int
    svg_valid_rate: float
    notes: list[str]


def build_report(model_artifacts: str, val_records: int) -> EvaluationReport:
    return EvaluationReport(
        model_artifacts=model_artifacts,
        val_records=val_records,
        svg_valid_rate=1.0 if val_records else 0.0,
        notes=[
            "Stub evaluation step",
            "Replace with a rendering-and-comparison metric once the inference loop is finalized",
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained SVG model")
    parser.add_argument("--model-artifacts", required=True, help="Path or S3 URI to model artifacts")
    parser.add_argument("--val-count", type=int, default=0, help="Number of validation records")
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation outputs")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(args.model_artifacts, args.val_count)
    output_path = output_dir / "evaluation.json"
    output_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))
    print(json.dumps(asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
