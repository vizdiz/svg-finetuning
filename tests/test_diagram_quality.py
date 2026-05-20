from __future__ import annotations

from backend.training.diagram_ir import (
    CanvasSpec,
    DiagramIRDocument,
    EdgeSpec,
    GroupSpec,
    LayoutSpec,
    NodeSpec,
    PortSpec,
    SizeHint,
)
from backend.training.diagram_quality import assess_compiled_svg, assess_diagram_ir


def _document() -> DiagramIRDocument:
    return DiagramIRDocument(
        diagram_type="architecture",
        title="Round trip",
        canvas=CanvasSpec(width=1000, height=600),
        nodes=[
            NodeSpec(
                id="client",
                kind="box",
                label="Client",
                size_hint=SizeHint(min_width=120, min_height=64),
                ports=[PortSpec(name="out", side="right")],
            ),
            NodeSpec(
                id="api",
                kind="box",
                label="API Gateway",
                ports=[PortSpec(name="in", side="left"), PortSpec(name="out", side="right")],
            ),
            NodeSpec(
                id="service",
                kind="box",
                label="Auth Service",
                ports=[PortSpec(name="in", side="left")],
            ),
        ],
        edges=[
            EdgeSpec(
                id="client-api",
                source="client",
                target="api",
                source_port="out",
                target_port="in",
            ),
            EdgeSpec(
                id="api-service",
                source="api",
                target="service",
                source_port="out",
                target_port="in",
                routing="orthogonal",
            ),
        ],
        groups=[GroupSpec(id="lane", members=["client", "api", "service"], label="Flow")],
        layout=LayoutSpec(direction="horizontal", alignment="center", spacing=48),
    )


def test_assess_diagram_ir_accepts_stable_document():
    report = assess_diagram_ir(_document())

    assert report.accepted is True
    assert report.reasons == []
    assert report.metrics["roundtrip_svg_stable"] is True
    assert report.metrics["signatures_match"] is True
    assert report.source_signature == report.parsed_signature


def test_assess_compiled_svg_rejects_malformed_render():
    svg = assess_diagram_ir(_document()).compiled_svg
    assert svg is not None

    report = assess_compiled_svg(svg[:-6])

    assert report.accepted is False
    assert any(reason.startswith("parse_failed") for reason in report.reasons)
