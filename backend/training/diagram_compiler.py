"""
training.diagram_compiler

Deterministic compiler from the diagram IR to SVG.

The compiler owns geometry:
  - node sizing
  - node placement
  - group frames
  - edge routing
  - label placement
  - arrow markers

The IR owns structure and intent. The compiler turns that into exact SVG
without relying on model-generated coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from xml.sax.saxutils import escape

from backend.training.diagram_ir import (
    CanvasSpec,
    DiagramIRDocument,
    EdgeSpec,
    GroupSpec,
    LayoutSpec,
    NodeSpec,
    PortSpec,
)

_SVG_NS = "http://www.w3.org/2000/svg"
_DEFAULT_FONT_SIZE = 12
_DEFAULT_FONT_FAMILY = "IBM Plex Mono"
_NODE_FILL = "#E8E7E2"
_NODE_STROKE = "#1A1A18"
_TEXT_COLOR = "#1A1A18"
_EDGE_COLOR = "#1A1A18"
_GROUP_FILL = "none"
_GROUP_STROKE = "#D0CFC8"


@dataclass(slots=True)
class NodeBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2


def _text_width(text: str, font_size: int = _DEFAULT_FONT_SIZE) -> float:
    if not text:
        return 0.0
    return max(len(line) for line in text.splitlines()) * font_size * 0.6


def _measure_node(node: NodeSpec) -> tuple[float, float]:
    font_size = node.style.text.font_size or _DEFAULT_FONT_SIZE
    lines = node.label.splitlines() if node.label else [node.id]
    longest = max(len(line) for line in lines) if lines else 0
    text_w = longest * font_size * 0.6
    text_h = max(1, len(lines)) * font_size * 1.5
    min_width = node.size_hint.min_width or 0
    min_height = node.size_hint.min_height or 0
    width = max(min_width, math.ceil(text_w + 32))
    height = max(min_height, math.ceil(text_h + 24))
    if node.size_hint.aspect_ratio:
        target = node.size_hint.aspect_ratio
        if width / height < target:
            width = math.ceil(height * target)
        else:
            height = math.ceil(width / target)
    return float(width), float(height)


def _port_position(box: NodeBox, port: PortSpec | None, fallback: str) -> tuple[float, float]:
    side = (port.side if port else fallback) or "auto"
    if side == "left":
        return box.left, box.cy
    if side == "right":
        return box.right, box.cy
    if side == "top":
        return box.cx, box.top
    if side == "bottom":
        return box.cx, box.bottom
    return box.cx, box.cy


def _layout_boxes(
    nodes: list[NodeSpec],
    canvas: CanvasSpec,
    layout: LayoutSpec,
) -> dict[str, NodeBox]:
    measured = {node.id: _measure_node(node) for node in nodes}
    boxes: dict[str, NodeBox] = {}
    padding = canvas.padding
    spacing = layout.spacing
    x = padding
    y = padding

    if layout.direction == "horizontal":
        row_height = 0.0
        for node in nodes:
            width, height = measured[node.id]
            boxes[node.id] = NodeBox(x=x, y=y, width=width, height=height)
            x += width + spacing
            row_height = max(row_height, height)
        return boxes

    if layout.direction == "vertical":
        col_width = 0.0
        for node in nodes:
            width, height = measured[node.id]
            boxes[node.id] = NodeBox(x=x, y=y, width=width, height=height)
            y += height + spacing
            col_width = max(col_width, width)
        return boxes

    # Freeform: preserve order but wrap if the row gets too wide.
    max_width = max(1.0, canvas.width - canvas.padding * 2)
    cursor_x = padding
    cursor_y = padding
    row_height = 0.0
    for node in nodes:
        width, height = measured[node.id]
        if cursor_x != padding and cursor_x + width > padding + max_width:
            cursor_x = padding
            cursor_y += row_height + spacing
            row_height = 0.0
        boxes[node.id] = NodeBox(x=cursor_x, y=cursor_y, width=width, height=height)
        cursor_x += width + spacing
        row_height = max(row_height, height)
    return boxes


def _group_boxes(groups: list[GroupSpec], boxes: dict[str, NodeBox]) -> dict[str, NodeBox]:
    frames: dict[str, NodeBox] = {}
    for group in groups:
        members = [boxes[node_id] for node_id in group.members if node_id in boxes]
        if not members:
            continue
        left = min(box.left for box in members) - group.padding
        top = min(box.top for box in members) - group.padding
        right = max(box.right for box in members) + group.padding
        bottom = max(box.bottom for box in members) + group.padding
        frames[group.id] = NodeBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        )
    return frames


def _edge_endpoints(
    edge: EdgeSpec,
    nodes: dict[str, NodeSpec],
    boxes: dict[str, NodeBox],
) -> tuple[tuple[float, float], tuple[float, float]]:
    source_node = nodes[edge.source]
    target_node = nodes[edge.target]
    source_box = boxes[edge.source]
    target_box = boxes[edge.target]
    source_port = next((port for port in source_node.ports if port.name == edge.source_port), None)
    target_port = next((port for port in target_node.ports if port.name == edge.target_port), None)

    source_side = source_port.side if source_port else ("right" if source_box.cx <= target_box.cx else "left")
    target_side = target_port.side if target_port else ("left" if source_box.cx <= target_box.cx else "right")

    return _port_position(source_box, source_port, source_side), _port_position(
        target_box, target_port, target_side
    )


def _route_points(
    edge: EdgeSpec,
    source: tuple[float, float],
    target: tuple[float, float],
) -> list[tuple[float, float]]:
    sx, sy = source
    tx, ty = target
    if edge.routing == "straight":
        return [source, target]
    if edge.routing == "curved":
        # SVG path is generated from these points, using a cubic midpoint bend.
        mid_x = (sx + tx) / 2
        return [source, (mid_x, sy), (mid_x, ty), target]

    # auto and orthogonal both use a simple elbow route.
    if abs(sx - tx) < 1e-6 or abs(sy - ty) < 1e-6:
        return [source, target]
    if abs(sx - tx) > abs(sy - ty):
        mid_x = (sx + tx) / 2
        return [source, (mid_x, sy), (mid_x, ty), target]
    mid_y = (sy + ty) / 2
    return [source, (sx, mid_y), (tx, mid_y), target]


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _path_from_points(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    if len(points) == 2:
        (sx, sy), (tx, ty) = points
        return f"M {sx:.1f} {sy:.1f} L {tx:.1f} {ty:.1f}"
    if len(points) == 4:
        (sx, sy), (c1x, c1y), (c2x, c2y), (tx, ty) = points
        return f"M {sx:.1f} {sy:.1f} C {c1x:.1f} {c1y:.1f}, {c2x:.1f} {c2y:.1f}, {tx:.1f} {ty:.1f}"
    commands = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    for x, y in points[1:]:
        commands.append(f"L {x:.1f} {y:.1f}")
    return " ".join(commands)


def _escape_label(label: str) -> str:
    return escape(label).replace("\n", "&#10;")


def _fmt_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    text = f"{float(value):.2f}"
    return text.rstrip("0").rstrip(".")


def compile_diagram_ir(document: DiagramIRDocument) -> str:
    document.validate()

    node_boxes = _layout_boxes(document.nodes, document.canvas, document.layout)
    group_boxes = _group_boxes(document.groups, node_boxes)
    node_map = document.node_map()
    width = document.canvas.width
    height = document.canvas.height
    bg = document.style.background or document.canvas.background
    line_color = document.style.line_color or _EDGE_COLOR
    node_fill = _NODE_FILL
    node_stroke = _NODE_STROKE
    text_color = document.style.foreground or _TEXT_COLOR
    group_stroke = _GROUP_STROKE

    parts: list[str] = [
        f'<svg xmlns="{_SVG_NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'fill="none" role="img" aria-label="{escape(document.title or document.diagram_type)}" '
        f'data-diagram-type="{escape(document.diagram_type)}" data-title="{escape(document.title)}" '
        f'data-layout-direction="{escape(document.layout.direction)}" '
        f'data-layout-spacing="{_fmt_number(document.layout.spacing)}" '
        f'data-layout-alignment="{escape(document.layout.alignment)}" '
        f'data-canvas-padding="{_fmt_number(document.canvas.padding)}" '
        f'data-canvas-grid="{_fmt_number(document.canvas.grid)}" '
        f'data-canvas-origin-x="{_fmt_number(document.canvas.origin_x)}" '
        f'data-canvas-origin-y="{_fmt_number(document.canvas.origin_y)}" '
        f'data-canvas-background="{escape(document.canvas.background)}">',
        f"<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"{bg}\"/>",
        "<defs>",
        (
            "<marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"10\" refX=\"8\" refY=\"3\" "
            "orient=\"auto\" markerUnits=\"strokeWidth\">"
            "<path d=\"M0,0 L0,6 L9,3 z\" fill=\"currentColor\"/></marker>"
        ),
        "</defs>",
    ]

    for group in document.groups:
        frame = group_boxes.get(group.id)
        if frame is None:
            continue
        parts.append(
            f'<g id="group-{escape(group.id)}">'
            f'<rect x="{frame.x:.1f}" y="{frame.y:.1f}" width="{frame.width:.1f}" height="{frame.height:.1f}" '
            f'fill="{group.style.fill or _GROUP_FILL}" stroke="{group.style.stroke or group_stroke}" '
            f'stroke-width="{_fmt_number(group.style.stroke_width or 1)}" />'
        )
        if group.label:
            parts.append(
                f'<text x="{frame.x + 12:.1f}" y="{frame.y + 18:.1f}" font-family="{_DEFAULT_FONT_FAMILY}" '
                f'font-size="11" fill="{text_color}">{_escape_label(group.label)}</text>'
            )
        parts.append("</g>")

    for node in document.nodes:
        box = node_boxes[node.id]
        parts.append(f'<g id="node-{escape(node.id)}">')
        parts.append(
            f'<rect x="{box.x:.1f}" y="{box.y:.1f}" width="{box.width:.1f}" height="{box.height:.1f}" '
            f'fill="{node.style.fill or node_fill}" stroke="{node.style.stroke or node_stroke}" '
            f'stroke-width="{_fmt_number(node.style.stroke_width or 1)}" />'
        )
        font_size = node.style.text.font_size or _DEFAULT_FONT_SIZE
        text_lines = node.label.splitlines() if node.label else [node.id]
        line_height = font_size * 1.35
        start_y = box.cy - ((len(text_lines) - 1) * line_height) / 2 - 0.35 * font_size
        text_anchor = node.style.text.align
        anchor = {"left": "start", "center": "middle", "right": "end"}[text_anchor]
        text_x = {
            "left": box.x + 16,
            "center": box.cx,
            "right": box.right - 16,
        }[text_anchor]
        style_attrs = []
        if node.style.text.italic:
            style_attrs.append('font-style="italic"')
        if node.style.text.weight:
            style_attrs.append(f'font-weight="{node.style.text.weight}"')
        if node.style.text.color or node.style.text.align:
            style_attrs.append(f'fill="{node.style.text.color or text_color}"')
        parts.append(
            f'<text x="{text_x:.1f}" y="{start_y:.1f}" text-anchor="{anchor}" '
            f'font-family="{_DEFAULT_FONT_FAMILY}" font-size="{font_size}" '
            f'{ " ".join(style_attrs) }>'
        )
        for index, line in enumerate(text_lines):
            dy = 0 if index == 0 else line_height
            tspan_attrs = f'x="{text_x:.1f}"' if index > 0 else ""
            parts.append(
                f'<tspan {tspan_attrs} dy="{dy:.1f}">{_escape_label(line)}</tspan>'
            )
        parts.append("</text>")
        parts.append("</g>")

    for edge in document.edges:
        source, target = _edge_endpoints(edge, node_map, node_boxes)
        points = _route_points(edge, source, target)
        edge_color = edge.style.stroke or line_color
        if edge.routing == "curved":
            path_d = _path_from_points(points)
            attrs = f'd="{path_d}" fill="none" stroke="{edge_color}" color="{edge_color}" '
        else:
            attrs = f'points="{_polyline(points)}" fill="none" stroke="{edge_color}" color="{edge_color}" '
        stroke_width = _fmt_number(edge.style.stroke_width or 1.5)
        dasharray = f' stroke-dasharray="{edge.style.dasharray}"' if edge.style.dasharray else ""
        marker_end = ' marker-end="url(#arrow)"' if edge.directed else ""
        parts.append(
            f'<g id="edge-{escape(edge.id)}">'
            + (
                f'<path {attrs}stroke-width="{stroke_width}"{dasharray}{marker_end} '
                f'stroke-linecap="round" stroke-linejoin="round" />'
                if edge.routing == "curved"
                else f'<polyline {attrs}stroke-width="{stroke_width}"{dasharray}{marker_end} '
                f'stroke-linecap="round" stroke-linejoin="round" />'
            )
        )
        if edge.label:
            mid_index = len(points) // 2
            lx, ly = points[mid_index]
            parts.append(
                f'<text x="{lx:.1f}" y="{ly - 6:.1f}" text-anchor="middle" '
                f'font-family="{_DEFAULT_FONT_FAMILY}" font-size="11" fill="{text_color}">'
                f"{_escape_label(edge.label)}</text>"
            )
        parts.append("</g>")

    parts.append("</svg>")
    return "".join(parts)
