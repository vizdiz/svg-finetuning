from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from backend.dataset_pipeline.processing.validator import validate_svg_detailed
from backend.training.diagram_compiler import compile_diagram_ir
from backend.training.diagram_ir import DiagramIRDocument


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_eval_report(
    *,
    records_path: Path,
    output_path: Path,
    min_schema_validity: float = 0.95,
    min_compilability: float = 0.90,
    min_render_validity: float = 0.95,
) -> dict[str, Any]:
    rows = _read_jsonl(records_path)
    total = len(rows)
    schema_valid = 0
    compilable = 0
    render_valid = 0
    node_counts: list[int] = []
    edge_counts: list[int] = []
    failures: list[dict[str, str]] = []

    for row in rows:
        diagram_ir = row.get("diagram_ir")
        if diagram_ir is not None:
            try:
                document = DiagramIRDocument.from_dict(diagram_ir)
                document.validate()
                schema_valid += 1
                node_counts.append(len(document.nodes))
                edge_counts.append(len(document.edges))
                compile_diagram_ir(document)
                compilable += 1
            except Exception as exc:
                failures.append({"id": str(row.get("id", "")), "stage": "ir", "reason": f"{type(exc).__name__}: {exc}"})
        if row.get("svg"):
            validation = validate_svg_detailed(row["svg"])
            if validation.ok:
                render_valid += 1
            else:
                failures.append({"id": str(row.get("id", "")), "stage": "svg", "reason": validation.reason})

    def rate(value: int, denominator: int = total) -> float:
        return value / denominator if denominator else 0.0

    report = {
        "record_count": total,
        "schema_validity_rate": rate(schema_valid),
        "compilability_rate": rate(compilable),
        "render_validity_rate": rate(render_valid, sum(1 for row in rows if row.get("svg"))),
        "avg_node_count": sum(node_counts) / len(node_counts) if node_counts else 0.0,
        "avg_edge_count": sum(edge_counts) / len(edge_counts) if edge_counts else 0.0,
        "passed": False,
        "thresholds": {
            "min_schema_validity": min_schema_validity,
            "min_compilability": min_compilability,
            "min_render_validity": min_render_validity,
        },
        "sample_failures": failures[:50],
    }
    report["passed"] = (
        report["schema_validity_rate"] >= min_schema_validity
        and report["compilability_rate"] >= min_compilability
        and report["render_validity_rate"] >= min_render_validity
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
