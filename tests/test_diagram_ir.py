from __future__ import annotations

import json

import pytest

from backend.training.diagram_ir import (
    CanvasSpec,
    DiagramIRDocument,
    EdgeSpec,
    GroupSpec,
    IRSchemaError,
    LayoutSpec,
    NodeStyle,
    NodeSpec,
    PortSpec,
    SizeHint,
    TextStyle,
)


def _document() -> DiagramIRDocument:
    return DiagramIRDocument(
        diagram_type="architecture",
        title="Auth flow",
        canvas=CanvasSpec(width=1200, height=800),
        nodes=[
            NodeSpec(
                id="client",
                kind="box",
                label="Client",
                size_hint=SizeHint(min_width=120, min_height=64),
                style=NodeStyle(
                    fill="#E8E7E2",
                    stroke="#1A1A18",
                    text=TextStyle(font_size=12, weight=400, italic=False, align="center"),
                ),
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
                label_position="mid",
            ),
        ],
        groups=[
            GroupSpec(id="lane", members=["client", "api", "service"], kind="swimlane")
        ],
        layout=LayoutSpec(direction="horizontal", alignment="center", spacing=48),
        metadata={"source": "synthetic"},
    )


def test_document_round_trips_through_json():
    doc = _document()
    raw = doc.to_json()

    restored = DiagramIRDocument.from_json(raw)

    assert restored.to_dict() == doc.to_dict()
    assert restored.schema_version == "0.1"
    assert restored.node_map()["client"].label == "Client"


def test_document_validation_rejects_unknown_references():
    doc = _document()
    doc.edges.append(EdgeSpec(id="bad", source="client", target="missing"))

    with pytest.raises(IRSchemaError, match="unknown target node"):
        doc.validate()


def test_document_validation_rejects_duplicate_ids():
    doc = _document()
    doc.nodes.append(NodeSpec(id="client", kind="box"))

    with pytest.raises(IRSchemaError, match="node ids must be unique"):
        doc.validate()


def test_document_validation_rejects_bad_port_reference():
    doc = _document()
    doc.edges[0].source_port = "missing"

    with pytest.raises(IRSchemaError, match="has no port named"):
        doc.validate()


def test_document_from_dict_rejects_bad_layout():
    payload = _document().to_dict()
    payload["layout"]["direction"] = "diagonal"

    with pytest.raises(IRSchemaError, match="layout.direction"):
        DiagramIRDocument.from_dict(payload)


def test_document_to_dict_is_json_compatible():
    payload = _document().to_dict()

    assert json.loads(json.dumps(payload)) == payload
