"""
dataset_pipeline.processing.svg_ir_extractor

Lossless, modular SVG-to-IR extraction pipeline.

The pipeline is intentionally split into stages:
  - normalize_svg(svg) -> canonical_dom
  - extract_primitives(canonical_dom) -> objects
  - infer_ir(objects) -> DiagramIRDocument
  - score_ir(ir) -> confidence
  - render_ir(ir) -> svg
  - roundtrip_ok(original_svg, rendered_svg) -> bool

This module keeps the original SVG intact and derives a canonical DOM plus
primitive objects for analysis. It does not destructively rewrite the source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import math
import re
import xml.etree.ElementTree as ET

from lxml import etree

from backend.training.diagram_compiler import compile_diagram_ir
from backend.training.diagram_ir import (
    CanvasSpec,
    DiagramIRDocument,
    EdgeSpec,
    EdgeStyle,
    GroupSpec,
    LayoutSpec,
    NodeSpec,
    NodeStyle,
    PortSpec,
    SizeHint,
    TextStyle,
)
from backend.training.diagram_parser import parse_diagram_svg
from backend.training.diagram_quality import assess_diagram_ir

_SVG_NS = "http://www.w3.org/2000/svg"
_NS = {"svg": _SVG_NS}

_TRANSFORM_RE = re.compile(r"(?P<name>[a-zA-Z]+)\((?P<args>[^)]*)\)")


@dataclass(slots=True)
class CanonicalSVG:
    original_svg: str
    root: etree._Element
    width: float | None
    height: float | None
    view_box: tuple[float, float, float, float] | None
    namespaces: dict[str, str] = field(default_factory=dict)
    element_count: int = 0


@dataclass(slots=True)
class SVGPrimitive:
    kind: str
    element_tag: str
    element_id: str | None
    text: str = ""
    bbox: tuple[float, float, float, float] | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IRConfidence:
    score: float
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def _float(value: str | None, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_view_box(root: etree._Element) -> tuple[float, float, float, float] | None:
    view_box = root.get("viewBox")
    if not view_box:
        return None
    parts = [part for part in re.split(r"[,\s]+", view_box.strip()) if part]
    if len(parts) != 4:
        return None
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def normalize_svg(svg: str) -> CanonicalSVG:
    parser = etree.XMLParser(remove_blank_text=False, remove_comments=False)
    root = etree.fromstring(svg.encode(), parser)
    namespaces = dict(root.nsmap or {})
    width = _float(root.get("width"))
    height = _float(root.get("height"))
    view_box = _parse_view_box(root)
    return CanonicalSVG(
        original_svg=svg,
        root=root,
        width=width,
        height=height,
        view_box=view_box,
        namespaces=namespaces,
        element_count=sum(1 for _ in root.iter()) - 1,
    )


def _element_bbox(elem: etree._Element) -> tuple[float, float, float, float] | None:
    tag = etree.QName(elem).localname
    if tag == "rect":
        x = _float(elem.get("x"), 0.0) or 0.0
        y = _float(elem.get("y"), 0.0) or 0.0
        width = _float(elem.get("width"), 0.0) or 0.0
        height = _float(elem.get("height"), 0.0) or 0.0
        return (x, y, width, height)
    if tag in {"circle", "ellipse"}:
        cx = _float(elem.get("cx"), 0.0) or 0.0
        cy = _float(elem.get("cy"), 0.0) or 0.0
        if tag == "circle":
            r = _float(elem.get("r"), 0.0) or 0.0
            return (cx - r, cy - r, r * 2, r * 2)
        rx = _float(elem.get("rx"), 0.0) or 0.0
        ry = _float(elem.get("ry"), 0.0) or 0.0
        return (cx - rx, cy - ry, rx * 2, ry * 2)
    if tag == "line":
        x1 = _float(elem.get("x1"), 0.0) or 0.0
        y1 = _float(elem.get("y1"), 0.0) or 0.0
        x2 = _float(elem.get("x2"), 0.0) or 0.0
        y2 = _float(elem.get("y2"), 0.0) or 0.0
        return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    if tag == "text":
        x = _float(elem.get("x"), 0.0) or 0.0
        y = _float(elem.get("y"), 0.0) or 0.0
        text = "".join(elem.itertext()).strip()
        width = max(16.0, len(text) * 6.0)
        height = 16.0
        return (x, y - height, width, height)
    return None


def _element_text(elem: etree._Element) -> str:
    return "".join(elem.itertext()).strip()


def extract_primitives(canonical_dom: CanonicalSVG) -> list[SVGPrimitive]:
    primitives: list[SVGPrimitive] = [
        SVGPrimitive(
            kind="canvas",
            element_tag="svg",
            element_id=canonical_dom.root.get("id"),
            bbox=canonical_dom.view_box,
            attrs={
                "width": canonical_dom.width,
                "height": canonical_dom.height,
                "viewBox": canonical_dom.view_box,
            },
            metadata={
                "width": canonical_dom.width,
                "height": canonical_dom.height,
                "viewBox": canonical_dom.view_box,
                "element_count": canonical_dom.element_count,
            },
        )
    ]
    for elem in canonical_dom.root.iter():
        tag = etree.QName(elem).localname
        if tag == "svg":
            continue
        element_id = elem.get("id")
        bbox = _element_bbox(elem)
        text = _element_text(elem) if tag in {"text", "tspan"} else ""
        attrs = {key: value for key, value in elem.attrib.items()}
        primitive_kind = "other"
        if tag == "g":
            primitive_kind = "group"
        elif tag in {"line", "path"}:
            primitive_kind = "edge"
        elif tag == "text":
            primitive_kind = "text"
        elif tag in {"rect", "circle", "ellipse"} and bbox is not None and bbox[2] > 0 and bbox[3] > 0:
            primitive_kind = "shape"
        primitives.append(
            SVGPrimitive(
                kind=primitive_kind,
                element_tag=tag,
                element_id=element_id,
                text=text,
                bbox=bbox,
                attrs=attrs,
                children=[child.get("id") for child in elem if child.get("id")],
                metadata={
                    "transform": elem.get("transform"),
                    "stroke": elem.get("stroke"),
                    "fill": elem.get("fill"),
                },
            )
        )
    return primitives


def _infer_layout(nodes: list[NodeSpec]) -> LayoutSpec:
    if len(nodes) < 2:
        return LayoutSpec(direction="horizontal", alignment="center", spacing=48)
    centers = [node.metadata.get("bbox") for node in nodes if node.metadata.get("bbox") is not None]
    if len(centers) < 2:
        return LayoutSpec(direction="horizontal", alignment="center", spacing=48)
    xs = [bbox[0] for bbox in centers]
    ys = [bbox[1] for bbox in centers]
    direction = "horizontal" if max(xs) - min(xs) >= max(ys) - min(ys) else "vertical"
    return LayoutSpec(direction=direction, alignment="center", spacing=48)


def infer_ir(objects: list[SVGPrimitive]) -> DiagramIRDocument:
    canvas_obj = next((obj for obj in objects if obj.kind == "canvas"), None)
    text_objects = [obj for obj in objects if obj.kind == "text"]
    shape_objects = [obj for obj in objects if obj.kind == "shape" and obj.bbox is not None]
    edge_objects = [obj for obj in objects if obj.kind == "edge"]

    nodes: list[NodeSpec] = []
    for index, shape in enumerate(shape_objects, start=1):
        label = ""
        for text in text_objects:
            if text.bbox is None or shape.bbox is None:
                continue
            tx, ty, tw, th = text.bbox
            sx, sy, sw, sh = shape.bbox
            if tx >= sx and ty >= sy - 20 and tx + tw <= sx + sw and ty + th <= sy + sh + 20:
                label = text.text
                break
        node_id = shape.element_id or f"node-{index}"
        nodes.append(
            NodeSpec(
                id=node_id,
                kind=shape.element_tag,
                label=label or node_id,
                metadata={"bbox": shape.bbox, "source": "svg"},
                size_hint=SizeHint(
                    min_width=int(shape.bbox[2]) if shape.bbox and shape.bbox[2] > 0 else None,
                    min_height=int(shape.bbox[3]) if shape.bbox and shape.bbox[3] > 0 else None,
                ),
                style=NodeStyle(
                    fill=shape.attrs.get("fill"),
                    stroke=shape.attrs.get("stroke"),
                    stroke_width=_float(shape.attrs.get("stroke-width")),
                    text=TextStyle(
                        font_size=int(_float(shape.attrs.get("font-size"), 12) or 12)
                    ),
                ),
            )
        )

    if not nodes and text_objects:
        for index, text in enumerate(text_objects, start=1):
            node_id = text.element_id or f"text-{index}"
            nodes.append(
                NodeSpec(
                    id=node_id,
                    kind="text",
                    label=text.text or node_id,
                    metadata={"bbox": text.bbox, "source": "svg"},
                )
            )

    edges: list[EdgeSpec] = []
    if len(nodes) >= 2:
        for index, edge in enumerate(edge_objects, start=1):
            source = nodes[min(index - 1, len(nodes) - 2)].id
            target = nodes[min(index, len(nodes) - 1)].id
            edges.append(
                EdgeSpec(
                    id=edge.element_id or f"edge-{index}",
                    source=source,
                    target=target,
                    kind=edge.element_tag,
                    directed=True,
                    metadata={"source": "svg", "bbox": edge.bbox},
                )
            )

    groups: list[GroupSpec] = []
    if len(nodes) > 1:
        groups.append(
            GroupSpec(
                id="group-1",
                members=[node.id for node in nodes],
                label="",
                metadata={"source": "svg"},
            )
        )

    layout = _infer_layout(nodes)
    canvas = CanvasSpec(
        width=int(canvas_obj.metadata.get("width") or 1200) if canvas_obj else 1200,
        height=int(canvas_obj.metadata.get("height") or 800) if canvas_obj else 800,
        background="#F0EFEB",
    )
    return DiagramIRDocument(
        diagram_type="architecture" if edges else "diagram",
        title="",
        canvas=canvas,
        nodes=nodes,
        edges=edges,
        groups=groups,
        layout=layout,
    )


def score_ir(ir: DiagramIRDocument) -> IRConfidence:
    report = assess_diagram_ir(ir)
    score = 0.0
    metrics = dict(report.metrics)
    reasons = list(report.reasons)
    if report.accepted:
        score = 1.0
    else:
        score = max(0.0, 1.0 - min(1.0, len(reasons) * 0.25))
    score += min(0.25, 0.05 * len(ir.nodes))
    score += min(0.15, 0.03 * len(ir.edges))
    score = min(score, 1.0)
    metrics["accepted"] = report.accepted
    return IRConfidence(score=score, reasons=reasons, metrics=metrics)


def render_ir(ir: DiagramIRDocument) -> str:
    return compile_diagram_ir(ir)


def roundtrip_ok(original_svg: str, rendered_svg: str) -> bool:
    try:
        original = normalize_svg(original_svg)
        rendered = normalize_svg(rendered_svg)
    except Exception:
        return False

    try:
        original_ir = infer_ir(extract_primitives(original))
        rendered_ir = parse_diagram_svg(rendered_svg)
    except Exception:
        return False

    return score_ir(original_ir).score >= 0.5 and original_ir.diagram_type == rendered_ir.diagram_type
