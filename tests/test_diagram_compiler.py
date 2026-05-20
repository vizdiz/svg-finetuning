from __future__ import annotations

import xml.etree.ElementTree as ET

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


NS = {"svg": "http://www.w3.org/2000/svg"}


def _document(direction: str = "horizontal") -> DiagramIRDocument:
    return DiagramIRDocument(
        diagram_type="flowchart",
        title="Compiler test",
        canvas=CanvasSpec(width=800, height=400),
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
        layout=LayoutSpec(direction=direction, alignment="center", spacing=48),
    )


def _parse(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def test_compile_document_produces_valid_svg():
    svg = compile_diagram_ir(_document())
    root = _parse(svg)

    assert root.tag.endswith("svg")
    assert root.attrib["width"] == "800"
    assert root.find("svg:defs", NS) is not None
    assert root.find(".//svg:marker", NS) is not None
    assert root.find(".//svg:g[@id='node-client']", NS) is not None
    assert root.find(".//svg:g[@id='group-lane']", NS) is not None


def test_compile_document_places_nodes_in_order():
    svg = compile_diagram_ir(_document())
    root = _parse(svg)

    rects = {
        elem.attrib["x"]: elem
        for elem in root.findall(".//svg:g[@id='node-client']/svg:rect", NS)
    }
    client_rect = root.find(".//svg:g[@id='node-client']/svg:rect", NS)
    api_rect = root.find(".//svg:g[@id='node-api']/svg:rect", NS)
    service_rect = root.find(".//svg:g[@id='node-service']/svg:rect", NS)

    assert client_rect is not None and api_rect is not None and service_rect is not None
    assert float(client_rect.attrib["x"]) < float(api_rect.attrib["x"]) < float(service_rect.attrib["x"])


def test_compile_document_emits_edge_geometry():
    svg = compile_diagram_ir(_document())
    root = _parse(svg)

    edge = root.find(".//svg:g[@id='edge-client-api']/svg:polyline", NS)
    assert edge is not None
    assert "marker-end" in edge.attrib
    assert edge.attrib["stroke"] == "#1A1A18"


def test_vertical_layout_changes_node_axis():
    svg = compile_diagram_ir(_document(direction="vertical"))
    root = _parse(svg)

    client_rect = root.find(".//svg:g[@id='node-client']/svg:rect", NS)
    api_rect = root.find(".//svg:g[@id='node-api']/svg:rect", NS)
    service_rect = root.find(".//svg:g[@id='node-service']/svg:rect", NS)

    assert client_rect is not None and api_rect is not None and service_rect is not None
    assert float(client_rect.attrib["y"]) < float(api_rect.attrib["y"]) < float(service_rect.attrib["y"])

