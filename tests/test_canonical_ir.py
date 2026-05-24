from __future__ import annotations

import json
import re

from backend.training.canonical_ir import canonicalize_diagram_ir, compact_ir_json
from backend.training.diagram_compiler import compile_diagram_ir
from backend.training.diagram_ir import (
    CanvasSpec,
    DiagramIRDocument,
    EdgeSpec,
    LayoutSpec,
    NodeSpec,
)


def _dense_document() -> DiagramIRDocument:
    nodes = [
        NodeSpec(
            id=f"text-{index}",
            kind="text",
            label=f"label {index}",
            metadata={"bbox": [10 + (index % 5) * 40, 20 + index * 12, 30, 10]},
        )
        for index in range(30)
    ]
    nodes.extend(
        [
            NodeSpec(id="shape-a", kind="rect", label="A", metadata={"bbox": [300, 80, 120, 60]}),
            NodeSpec(id="shape-b", kind="rect", label="B", metadata={"bbox": [480, 80, 120, 60]}),
        ]
    )
    return DiagramIRDocument(
        diagram_type="figure",
        title="Dense",
        canvas=CanvasSpec(width=800, height=600),
        nodes=nodes,
        edges=[EdgeSpec(id="shape-edge", source="shape-a", target="shape-b")],
        layout=LayoutSpec(direction="horizontal"),
    )


def test_canonicalize_diagram_ir_aggregates_dense_text_and_preserves_geometry():
    canonical, report = canonicalize_diagram_ir(_dense_document(), max_nodes=10, max_edges=8)

    assert len(canonical.nodes) <= 10
    assert report.text_nodes_aggregated > 0
    assert report.source_nodes == 32
    assert report.canonical_edges == 1
    assert canonical.layout.direction == "freeform"
    assert all("bbox" in node.metadata for node in canonical.nodes)
    assert len(compact_ir_json(canonical)) < len(json.dumps(_dense_document().to_dict()))


def test_compiler_uses_freeform_canonical_bboxes():
    canonical, _ = canonicalize_diagram_ir(_dense_document(), max_nodes=10, max_edges=8)
    svg = compile_diagram_ir(canonical)

    assert re.search(r'<g id="node-s001">\s*<rect x="300', svg)
    assert re.search(r'<g id="node-s002">\s*<rect x="480', svg)
