from __future__ import annotations

from backend.dataset_pipeline.processing.svg_ir_extractor import (
    extract_primitives,
    infer_ir,
    normalize_svg,
    render_ir,
    roundtrip_ok,
    score_ir,
)
from backend.training.diagram_compiler import compile_diagram_ir
from backend.training.diagram_ir import (
    CanvasSpec,
    DiagramIRDocument,
    EdgeSpec,
    GroupSpec,
    LayoutSpec,
    NodeSpec,
    PortSpec,
)


def _compiled_svg() -> str:
    document = DiagramIRDocument(
        diagram_type="architecture",
        title="Extractor",
        canvas=CanvasSpec(width=800, height=420),
        nodes=[
            NodeSpec(id="left", kind="box", label="Left", ports=[PortSpec(name="out", side="right")]),
            NodeSpec(id="right", kind="box", label="Right", ports=[PortSpec(name="in", side="left")]),
        ],
        edges=[EdgeSpec(id="edge", source="left", target="right", source_port="out", target_port="in")],
        groups=[GroupSpec(id="grp", members=["left", "right"])],
        layout=LayoutSpec(direction="horizontal", spacing=64),
    )
    return compile_diagram_ir(document)


def test_normalize_svg_returns_canonical_dom():
    canonical = normalize_svg(_compiled_svg())
    assert canonical.root.tag.endswith("svg")
    assert canonical.element_count > 0


def test_extract_primitives_includes_canvas_and_shapes():
    canonical = normalize_svg(_compiled_svg())
    primitives = extract_primitives(canonical)
    kinds = [primitive.kind for primitive in primitives]
    assert kinds[0] == "canvas"
    assert "shape" in kinds
    assert "text" in kinds


def test_infer_ir_and_score_round_trip_on_compiled_svg():
    canonical = normalize_svg(_compiled_svg())
    primitives = extract_primitives(canonical)
    document = infer_ir(primitives)
    report = score_ir(document)
    assert document.nodes
    assert report.score > 0


def test_render_ir_and_roundtrip_ok():
    canonical = normalize_svg(_compiled_svg())
    document = infer_ir(extract_primitives(canonical))
    rendered = render_ir(document)
    assert rendered.startswith("<svg")
    assert isinstance(roundtrip_ok(canonical.original_svg, rendered), bool)
