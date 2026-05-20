from __future__ import annotations

from backend.training.diagram_compiler import compile_diagram_ir
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
from backend.training.diagram_parser import parse_diagram_svg


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


def test_parse_compiler_svg_recovers_structure():
    source = _document()
    svg = compile_diagram_ir(source)

    parsed = parse_diagram_svg(svg)

    assert parsed.canvas.width == source.canvas.width
    assert parsed.canvas.height == source.canvas.height
    assert parsed.layout.direction == "horizontal"
    assert parsed.layout.spacing == 48
    assert [node.id for node in parsed.nodes] == ["client", "api", "service"]
    assert [node.label for node in parsed.nodes] == ["Client", "API Gateway", "Auth Service"]
    assert [edge.id for edge in parsed.edges] == ["client-api", "api-service"]
    assert parsed.edges[0].source == "client"
    assert parsed.edges[0].target == "api"
    assert parsed.edges[0].source_port in {"right", "right_2", "out"}
    assert parsed.groups[0].id == "lane"
    assert parsed.groups[0].members == ["client", "api", "service"]


def test_parse_then_compile_is_stable():
    source = _document()
    svg = compile_diagram_ir(source)

    parsed = parse_diagram_svg(svg)
    recompilation = compile_diagram_ir(parsed)

    assert recompilation == svg

