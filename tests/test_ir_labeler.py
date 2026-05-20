from __future__ import annotations

from types import SimpleNamespace

from backend.dataset_pipeline.processing.ir_labeler import label_raw_svg_ir
from backend.dataset_pipeline.scrapers.base_scraper import RawSVG
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


def _svg() -> str:
    doc = DiagramIRDocument(
        diagram_type="architecture",
        title="IR label",
        canvas=CanvasSpec(width=900, height=500),
        nodes=[
            NodeSpec(id="a", kind="box", label="A", ports=[PortSpec(name="out", side="right")]),
            NodeSpec(id="b", kind="box", label="B", ports=[PortSpec(name="in", side="left")]),
        ],
        edges=[EdgeSpec(id="a-b", source="a", target="b", source_port="out", target_port="in")],
        groups=[GroupSpec(id="lane", members=["a", "b"], label="Flow")],
        layout=LayoutSpec(direction="horizontal", alignment="center", spacing=48),
    )
    return compile_diagram_ir(doc)


def test_label_raw_svg_ir_attaches_metadata():
    raw = RawSVG(
        svg_string=_svg(),
        source_url="https://example.com/a.svg",
        source_id="a",
        domain="arxiv",
        metadata={},
    )

    result = label_raw_svg_ir(raw)

    assert result.accepted is True
    assert raw.metadata["diagram_ir_schema_version"] == "0.1"
    assert raw.metadata["diagram_ir_accepted"] is True
    assert raw.metadata["diagram_ir"]["diagram_type"] == "architecture"
    assert raw.metadata["diagram_ir"]["nodes"]
    assert raw.metadata["diagram_ir_confidence"] > 0


def test_label_raw_svg_ir_rejects_non_compiler_svg():
    raw = RawSVG(
        svg_string="<svg><rect /></svg>",
        source_url="https://example.com/b.svg",
        source_id="b",
        domain="arxiv",
        metadata={},
    )

    result = label_raw_svg_ir(raw)

    assert result.accepted is False
    assert result.reason == "empty_ir_document"
    assert raw.metadata == {}


def test_label_raw_svg_ir_normalizes_before_parse(monkeypatch):
    from backend.dataset_pipeline.processing import ir_labeler

    captured = {}

    def fake_normalize(svg: str) -> str:
        captured["normalize_input"] = svg
        return SimpleNamespace(original_svg="<svg normalized='1'/>")

    def fake_extract(canonical_dom):
        captured["extract_input"] = canonical_dom.original_svg
        return ["primitive"]

    def fake_infer(objects):
        captured["infer_input"] = list(objects)
        return SimpleNamespace(
            nodes=[SimpleNamespace(id="n1")],
            edges=[],
            groups=[],
            diagram_type="diagram",
            schema_version="0.1",
            to_dict=lambda: {"diagram_type": "diagram"},
        )

    def fake_score(document):
        captured["score_input"] = document.diagram_type
        return SimpleNamespace(score=0.9, metrics={"ok": True}, reasons=[])

    def fake_render(document):
        captured["render_input"] = document.diagram_type
        return "<svg rendered='1'/>"

    def fake_roundtrip(original_svg: str, rendered_svg: str) -> bool:
        captured["roundtrip_input"] = (original_svg, rendered_svg)
        return True

    monkeypatch.setattr(ir_labeler, "normalize_svg", fake_normalize)
    monkeypatch.setattr(ir_labeler, "extract_primitives", fake_extract)
    monkeypatch.setattr(ir_labeler, "infer_ir", fake_infer)
    monkeypatch.setattr(ir_labeler, "score_ir", fake_score)
    monkeypatch.setattr(ir_labeler, "render_ir", fake_render)
    monkeypatch.setattr(ir_labeler, "roundtrip_ok", fake_roundtrip)

    raw = RawSVG(
        svg_string="<svg raw='1'/>",
        source_url="https://example.com/c.svg",
        source_id="c",
        domain="arxiv",
        metadata={},
    )

    result = label_raw_svg_ir(raw)

    assert result.accepted is True
    assert captured["normalize_input"] == "<svg raw='1'/>"
    assert captured["extract_input"] == "<svg normalized='1'/>"
    assert captured["infer_input"] == ["primitive"]
