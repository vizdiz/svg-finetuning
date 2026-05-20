"""
training.diagram_quality

Structural quality gate for diagram IR and compiler output.

This gate is intentionally deterministic:
  - validate the IR
  - compile it to SVG
  - parse the SVG back into IR
  - compare structural signatures
  - verify the compiler output is stable under round-trip

It does not do pixel metrics. The goal is to reject malformed or
structurally unstable diagrams before they can reach training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

from backend.training.diagram_compiler import compile_diagram_ir
from backend.training.diagram_ir import DiagramIRDocument
from backend.training.diagram_parser import parse_diagram_svg


@dataclass(slots=True)
class DiagramQualityReport:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    source_signature: dict[str, Any] | None = None
    parsed_signature: dict[str, Any] | None = None
    compiled_svg: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _port_side(document: DiagramIRDocument, node_id: str, port_name: str | None) -> str | None:
    if port_name is None:
        return None
    node = next((item for item in document.nodes if item.id == node_id), None)
    if node is None:
        return None
    port = next((item for item in node.ports if item.name == port_name), None)
    return port.side if port is not None else None


def _document_signature(document: DiagramIRDocument) -> dict[str, Any]:
    return {
        "schema_version": document.schema_version,
        "diagram_type": document.diagram_type,
        "title": document.title,
        "canvas": {
            "width": document.canvas.width,
            "height": document.canvas.height,
            "grid": document.canvas.grid,
            "origin_x": document.canvas.origin_x,
            "origin_y": document.canvas.origin_y,
            "padding": document.canvas.padding,
            "background": document.canvas.background,
        },
        "layout": {
            "direction": document.layout.direction,
            "alignment": document.layout.alignment,
            "spacing": document.layout.spacing,
            "node_spacing": document.layout.node_spacing,
            "rank_spacing": document.layout.rank_spacing,
            "snap_grid": document.layout.snap_grid,
            "text_wrap": document.layout.text_wrap,
            "edge_routing": document.layout.edge_routing,
        },
        "nodes": [
            {
                "id": node.id,
                "kind": node.kind,
                "label": node.label,
                "role": node.role,
                "group_id": node.group_id,
                "ports": [
                    {
                        "side": port.side,
                    }
                    for port in node.ports
                ],
            }
            for node in document.nodes
        ],
        "edges": [
            {
                "id": edge.id,
                "source": edge.source,
                "target": edge.target,
                "kind": edge.kind,
                "directed": edge.directed,
                "label": edge.label,
                "source_port_side": _port_side(document, edge.source, edge.source_port),
                "target_port_side": _port_side(document, edge.target, edge.target_port),
            }
            for edge in document.edges
        ],
        "groups": [
            {
                "id": group.id,
                "members": list(group.members),
                "kind": group.kind,
                "label": group.label,
            }
            for group in document.groups
        ],
    }


def assess_diagram_ir(document: DiagramIRDocument) -> DiagramQualityReport:
    reasons: list[str] = []
    metrics: dict[str, Any] = {
        "node_count": len(document.nodes),
        "edge_count": len(document.edges),
        "group_count": len(document.groups),
    }

    try:
        document.validate()
    except Exception as exc:
        reasons.append(f"invalid_ir: {type(exc).__name__}: {exc}")
        return DiagramQualityReport(
            accepted=False,
            reasons=reasons,
            metrics=metrics,
            source_signature=_document_signature(document),
        )

    try:
        compiled_svg = compile_diagram_ir(document)
    except Exception as exc:
        reasons.append(f"compile_failed: {type(exc).__name__}: {exc}")
        return DiagramQualityReport(
            accepted=False,
            reasons=reasons,
            metrics=metrics,
            source_signature=_document_signature(document),
        )

    try:
        parsed = parse_diagram_svg(compiled_svg)
    except Exception as exc:
        reasons.append(f"parse_failed: {type(exc).__name__}: {exc}")
        return DiagramQualityReport(
            accepted=False,
            reasons=reasons,
            metrics=metrics,
            source_signature=_document_signature(document),
            compiled_svg=compiled_svg,
        )

    try:
        recompilation = compile_diagram_ir(parsed)
    except Exception as exc:
        reasons.append(f"recompile_failed: {type(exc).__name__}: {exc}")
        return DiagramQualityReport(
            accepted=False,
            reasons=reasons,
            metrics=metrics,
            source_signature=_document_signature(document),
            parsed_signature=_document_signature(parsed),
            compiled_svg=compiled_svg,
        )

    source_signature = _document_signature(document)
    parsed_signature = _document_signature(parsed)
    metrics.update(
        {
            "roundtrip_svg_stable": recompilation == compiled_svg,
            "signatures_match": source_signature == parsed_signature,
        }
    )

    if recompilation != compiled_svg:
        reasons.append("roundtrip_svg_mismatch")
    if source_signature != parsed_signature:
        reasons.append("structural_signature_mismatch")

    accepted = not reasons
    return DiagramQualityReport(
        accepted=accepted,
        reasons=reasons,
        metrics=metrics,
        source_signature=source_signature,
        parsed_signature=parsed_signature,
        compiled_svg=compiled_svg,
    )


def assess_compiled_svg(svg: str) -> DiagramQualityReport:
    reasons: list[str] = []
    try:
        parsed = parse_diagram_svg(svg)
    except Exception as exc:
        return DiagramQualityReport(
            accepted=False,
            reasons=[f"parse_failed: {type(exc).__name__}: {exc}"],
            metrics={},
        )

    try:
        recompilation = compile_diagram_ir(parsed)
    except Exception as exc:
        return DiagramQualityReport(
            accepted=False,
            reasons=[f"recompile_failed: {type(exc).__name__}: {exc}"],
            metrics={
                "node_count": len(parsed.nodes),
                "edge_count": len(parsed.edges),
                "group_count": len(parsed.groups),
            },
            parsed_signature=_document_signature(parsed),
            compiled_svg=svg,
        )

    metrics = {
        "node_count": len(parsed.nodes),
        "edge_count": len(parsed.edges),
        "group_count": len(parsed.groups),
        "roundtrip_svg_stable": recompilation == svg,
    }
    if recompilation != svg:
        reasons.append("roundtrip_svg_mismatch")

    return DiagramQualityReport(
        accepted=not reasons,
        reasons=reasons,
        metrics=metrics,
        parsed_signature=_document_signature(parsed),
        compiled_svg=svg,
    )
