"""
training.diagram_parser

Inverse parser for the SVG emitted by training.diagram_compiler.

This is intentionally conservative. It is not a generic SVG parser.
It reconstructs the narrow diagram IR from the compiler's SVG dialect so
we can run round-trip checks and structural validation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from statistics import median
from typing import Any
import xml.etree.ElementTree as ET

from backend.training.diagram_ir import (
    CanvasSpec,
    DiagramIRDocument,
    EdgeSpec,
    EdgeStyle,
    GroupSpec,
    GroupStyle,
    LayoutSpec,
    NodeSpec,
    NodeStyle,
    PortSpec,
    SizeHint,
    TextStyle,
)

_SVG_NS = "http://www.w3.org/2000/svg"
_NS = {"svg": _SVG_NS}
_DEFAULT_FONT_SIZE = 12
_EDGE_ROUTING_MAP = {"path": "curved", "polyline": "orthogonal"}


@dataclass(slots=True)
class ParsedNode:
    node: NodeSpec
    box: tuple[float, float, float, float]
    group_id: str | None = None


@dataclass(slots=True)
class ParsedEdge:
    edge: EdgeSpec
    source_point: tuple[float, float]
    target_point: tuple[float, float]


def _float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rect_attrs(elem: ET.Element) -> tuple[float, float, float, float]:
    return (
        _float(elem.attrib.get("x")),
        _float(elem.attrib.get("y")),
        _float(elem.attrib.get("width")),
        _float(elem.attrib.get("height")),
    )


def _parse_text(elem: ET.Element) -> str:
    tspans = elem.findall("svg:tspan", _NS)
    if tspans:
        return "\n".join("".join(tspan.itertext()) for tspan in tspans)
    return "".join(elem.itertext()).strip()


def _parse_text_style(elem: ET.Element) -> TextStyle:
    font_size = elem.attrib.get("font-size")
    font_weight = elem.attrib.get("font-weight")
    font_style = elem.attrib.get("font-style")
    text_anchor = elem.attrib.get("text-anchor", "middle")
    align = {"start": "left", "middle": "center", "end": "right"}.get(text_anchor, "center")
    return TextStyle(
        font_size=int(float(font_size)) if font_size else None,
        weight=int(float(font_weight)) if font_weight else None,
        italic=font_style == "italic",
        align=align,
        color=elem.attrib.get("fill"),
    )


def _parse_node(group: ET.Element) -> ParsedNode | None:
    group_id = group.attrib.get("id", "")
    if not group_id.startswith("node-"):
        return None
    node_id = group_id[len("node-") :]
    rect = group.find("svg:rect", _NS)
    text = group.find("svg:text", _NS)
    if rect is None:
        return None
    x, y, width, height = _rect_attrs(rect)
    label = _parse_text(text) if text is not None else node_id
    style = NodeStyle(
        fill=rect.attrib.get("fill"),
        stroke=rect.attrib.get("stroke"),
        stroke_width=_float(rect.attrib.get("stroke-width")) if rect.attrib.get("stroke-width") else None,
        text=_parse_text_style(text) if text is not None else TextStyle(),
    )
    node = NodeSpec(
        id=node_id,
        kind="box",
        label=label,
        size_hint=SizeHint(min_width=int(round(width)), min_height=int(round(height))),
        style=style,
        metadata={},
    )
    return ParsedNode(node=node, box=(x, y, width, height))


def _parse_points(points: str) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    for chunk in points.split():
        if "," not in chunk:
            continue
        x_str, y_str = chunk.split(",", 1)
        parsed.append((float(x_str), float(y_str)))
    return parsed


_PATH_TOKEN_RE = re.compile(r"[MLC]|-?\d+(?:\.\d+)?")


def _parse_path(points: str) -> list[tuple[float, float]]:
    tokens = _PATH_TOKEN_RE.findall(points)
    coords: list[float] = []
    for token in tokens:
        if token in {"M", "L", "C"}:
            continue
        coords.append(float(token))
    parsed: list[tuple[float, float]] = []
    for i in range(0, len(coords) - 1, 2):
        parsed.append((coords[i], coords[i + 1]))
    return parsed


def _closest_side(point: tuple[float, float], box: tuple[float, float, float, float]) -> str:
    px, py = point
    x, y, width, height = box
    left = abs(px - x)
    right = abs(px - (x + width))
    top = abs(py - y)
    bottom = abs(py - (y + height))
    distances = {"left": left, "right": right, "top": top, "bottom": bottom}
    return min(distances, key=distances.get)


def _nearest_node(point: tuple[float, float], nodes: list[ParsedNode]) -> ParsedNode:
    def score(candidate: ParsedNode) -> float:
        x, y, width, height = candidate.box
        px, py = point
        dx = max(x - px, 0.0, px - (x + width))
        dy = max(y - py, 0.0, py - (y + height))
        return math.hypot(dx, dy)

    return min(nodes, key=score)


def _node_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, width, height = box
    return (x + width / 2, y + height / 2)


def _assign_port_name(side: str, counts: Counter[str]) -> str:
    counts[side] += 1
    if counts[side] == 1:
        return side
    return f"{side}_{counts[side]}"


def _infer_layout(nodes: list[ParsedNode]) -> tuple[str, int]:
    if len(nodes) < 2:
        return "horizontal", 48

    xs = [parsed.box[0] for parsed in nodes]
    ys = [parsed.box[1] for parsed in nodes]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    if x_span >= y_span:
        gaps = [
            nodes[i + 1].box[0] - (nodes[i].box[0] + nodes[i].box[2])
            for i in range(len(nodes) - 1)
            if nodes[i + 1].box[0] >= nodes[i].box[0]
        ]
        spacing = int(round(median(gaps))) if gaps else 48
        return "horizontal", max(1, spacing)

    gaps = [
        nodes[i + 1].box[1] - (nodes[i].box[1] + nodes[i].box[3])
        for i in range(len(nodes) - 1)
        if nodes[i + 1].box[1] >= nodes[i].box[1]
    ]
    spacing = int(round(median(gaps))) if gaps else 48
    return "vertical", max(1, spacing)


def _parse_groups(root: ET.Element, parsed_nodes: list[ParsedNode]) -> list[GroupSpec]:
    groups: list[GroupSpec] = []
    for group_elem in root.findall("svg:g", _NS):
        group_id = group_elem.attrib.get("id", "")
        if not group_id.startswith("group-"):
            continue
        raw_group_id = group_id[len("group-") :]
        rect = group_elem.find("svg:rect", _NS)
        if rect is None:
            continue
        x, y, width, height = _rect_attrs(rect)
        label_elem = group_elem.find("svg:text", _NS)
        label = _parse_text(label_elem) if label_elem is not None else ""
        members = []
        for parsed_node in parsed_nodes:
            nx, ny, nw, nh = parsed_node.box
            if nx >= x and ny >= y and nx + nw <= x + width and ny + nh <= y + height:
                members.append(parsed_node.node.id)
                parsed_node.group_id = raw_group_id
        groups.append(
            GroupSpec(
                id=raw_group_id,
                members=members,
                kind="cluster",
                label=label,
                style=GroupStyle(
                    fill=rect.attrib.get("fill"),
                    stroke=rect.attrib.get("stroke"),
                    stroke_width=_float(rect.attrib.get("stroke-width")) if rect.attrib.get("stroke-width") else None,
                ),
            )
        )
    return groups


def _parse_edges(root: ET.Element, parsed_nodes: list[ParsedNode]) -> tuple[list[ParsedEdge], dict[str, Counter[str]]]:
    nodes_by_id = {parsed.node.id: parsed for parsed in parsed_nodes}
    port_usage: dict[str, Counter[str]] = defaultdict(Counter)
    edges: list[ParsedEdge] = []

    for edge_elem in root.findall("svg:g", _NS):
        edge_id = edge_elem.attrib.get("id", "")
        if not edge_id.startswith("edge-"):
            continue
        raw_edge_id = edge_id[len("edge-") :]
        geom = edge_elem.find("svg:path", _NS)
        if geom is None:
            geom = edge_elem.find("svg:polyline", _NS)
        if geom is None:
            continue
        directed = geom.attrib.get("marker-end") is not None
        stroke = geom.attrib.get("stroke")
        dasharray = geom.attrib.get("stroke-dasharray")
        stroke_width = _float(geom.attrib.get("stroke-width")) if geom.attrib.get("stroke-width") else None
        label_elem = edge_elem.find("svg:text", _NS)
        label = _parse_text(label_elem) if label_elem is not None else ""
        if geom.tag.endswith("path"):
            points = _parse_path(geom.attrib.get("d", ""))
            routing = "curved"
        else:
            points = _parse_points(geom.attrib.get("points", ""))
            routing = "orthogonal" if len(points) > 2 else "straight"
        if len(points) < 2:
            continue
        source_point = points[0]
        target_point = points[-1]
        source_node = _nearest_node(source_point, parsed_nodes)
        target_node = _nearest_node(target_point, parsed_nodes)
        source_side = _closest_side(source_point, source_node.box)
        target_side = _closest_side(target_point, target_node.box)
        source_port = _assign_port_name(source_side, port_usage[source_node.node.id])
        target_port = _assign_port_name(target_side, port_usage[target_node.node.id])
        source_node.node.ports.append(PortSpec(name=source_port, side=source_side))
        target_node.node.ports.append(PortSpec(name=target_port, side=target_side))
        edge = EdgeSpec(
            id=raw_edge_id,
            source=source_node.node.id,
            target=target_node.node.id,
            kind="link",
            directed=directed,
            label=label,
            routing=routing,
            label_position="mid",
            source_port=source_port,
            target_port=target_port,
            style=EdgeStyle(
                stroke=stroke,
                stroke_width=stroke_width,
                dasharray=dasharray,
                arrowhead="arrow" if directed else None,
            ),
            metadata={},
        )
        edges.append(ParsedEdge(edge=edge, source_point=source_point, target_point=target_point))
    return edges, port_usage


def parse_diagram_svg(svg: str) -> DiagramIRDocument:
    root = ET.fromstring(svg)
    if root.tag != f"{{{_SVG_NS}}}svg" and not root.tag.endswith("svg"):
        raise ValueError("expected an SVG root element")

    width = int(round(_float(root.attrib.get("width"), 1200)))
    height = int(round(_float(root.attrib.get("height"), 800)))
    canvas = CanvasSpec(
        width=width,
        height=height,
        grid=int(round(_float(root.attrib.get("data-canvas-grid"), 8))),
        origin_x=int(round(_float(root.attrib.get("data-canvas-origin-x"), 0))),
        origin_y=int(round(_float(root.attrib.get("data-canvas-origin-y"), 0))),
        padding=int(round(_float(root.attrib.get("data-canvas-padding"), 32))),
        background=root.attrib.get("data-canvas-background", "#F0EFEB"),
    )

    bg_rect = root.find("svg:rect", _NS)
    if bg_rect is not None and bg_rect.attrib.get("fill"):
        canvas.background = bg_rect.attrib["fill"]

    parsed_nodes: list[ParsedNode] = []
    for group in root.findall("svg:g", _NS):
        parsed = _parse_node(group)
        if parsed is not None:
            parsed_nodes.append(parsed)

    groups = _parse_groups(root, parsed_nodes)
    edges, _ = _parse_edges(root, parsed_nodes)

    if root.attrib.get("data-layout-direction"):
        layout_direction = root.attrib["data-layout-direction"]
    elif parsed_nodes:
        layout_direction, _ = _infer_layout(parsed_nodes)
    else:
        layout_direction = "horizontal"
    if root.attrib.get("data-layout-spacing"):
        spacing = int(round(_float(root.attrib.get("data-layout-spacing"), 48)))
    elif parsed_nodes:
        _, spacing = _infer_layout(parsed_nodes)
    else:
        spacing = 48
    alignment = root.attrib.get("data-layout-alignment", "center")

    nodes = [parsed.node for parsed in parsed_nodes]
    if any(node.style.text.color for node in nodes):
        text_color = next((node.style.text.color for node in nodes if node.style.text.color), None)
    else:
        text_color = None
    if edges:
        line_color = next((edge.edge.style.stroke for edge in edges if edge.edge.style.stroke), None)
    else:
        line_color = None

    layout = LayoutSpec(direction=layout_direction, alignment=alignment, spacing=spacing)
    document = DiagramIRDocument(
        diagram_type=root.attrib.get("data-diagram-type", root.attrib.get("aria-label", "flowchart")),
        title=root.attrib.get("data-title", root.attrib.get("aria-label", "")),
        canvas=canvas,
        nodes=nodes,
        edges=[parsed.edge for parsed in edges],
        groups=groups,
        layout=layout,
        metadata={},
    )
    if text_color or line_color:
        document.style.foreground = text_color
        document.style.line_color = line_color
    document.validate()
    return document
