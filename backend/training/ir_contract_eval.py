from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

from backend.training.diagram_compiler import compile_diagram_ir
from backend.training.diagram_ir import DiagramIRDocument


@dataclass(slots=True)
class IRContractScore:
    id: str
    schema_valid: bool
    compilable: bool
    label_recall: float
    edge_recall: float
    missing_labels: list[str] = field(default_factory=list)
    missing_edges: list[dict[str, str]] = field(default_factory=list)
    flaws: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip().replace("<|im_end|>", "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text[:-3]
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _rate(found: int, total: int) -> float:
    return found / total if total else 1.0


def score_ir_contract(record: dict[str, Any], raw_generation: str) -> IRContractScore:
    record_id = str(record.get("id") or "")
    expected_labels = [str(value) for value in (record.get("metadata") or {}).get("expected_labels", [])]
    expected_edges = list((record.get("metadata") or {}).get("expected_edges", []))
    flaws: list[str] = []

    payload = _extract_json_object(raw_generation)
    if payload is None:
        return IRContractScore(
            id=record_id,
            schema_valid=False,
            compilable=False,
            label_recall=0.0,
            edge_recall=0.0,
            missing_labels=expected_labels,
            missing_edges=expected_edges,
            flaws=["json_parse_failed"],
        )

    try:
        document = DiagramIRDocument.from_dict(payload)
        schema_valid = True
    except Exception as exc:
        return IRContractScore(
            id=record_id,
            schema_valid=False,
            compilable=False,
            label_recall=0.0,
            edge_recall=0.0,
            missing_labels=expected_labels,
            missing_edges=expected_edges,
            flaws=[f"schema_invalid:{type(exc).__name__}"],
        )

    try:
        compile_diagram_ir(document)
        compilable = True
    except Exception as exc:
        compilable = False
        flaws.append(f"compile_failed:{type(exc).__name__}")

    actual_labels = {node.label for node in document.nodes}
    missing_labels = [label for label in expected_labels if label not in actual_labels]

    node_labels = {node.id: node.label for node in document.nodes}
    actual_edges = {
        (node_labels.get(edge.source, edge.source), node_labels.get(edge.target, edge.target))
        for edge in document.edges
    }
    missing_edges = [
        edge
        for edge in expected_edges
        if (str(edge.get("source")), str(edge.get("target"))) not in actual_edges
    ]

    if missing_labels:
        flaws.append("missing_labels")
    if missing_edges:
        flaws.append("missing_edges")

    return IRContractScore(
        id=record_id,
        schema_valid=schema_valid,
        compilable=compilable,
        label_recall=_rate(len(expected_labels) - len(missing_labels), len(expected_labels)),
        edge_recall=_rate(len(expected_edges) - len(missing_edges), len(expected_edges)),
        missing_labels=missing_labels,
        missing_edges=missing_edges,
        flaws=flaws,
    )
