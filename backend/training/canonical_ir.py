"""
training.canonical_ir

Compacts extracted SVG-derived IR into the compiler's canonical diagram IR.

The extractor can preserve hundreds of text fragments and thousands of raw
path-like edges. That is valuable source evidence, but it is too large and too
noisy as a language-model target. Canonical IR keeps geometric anchors and
visible structure while aggregating dense fragments into compiler-sized nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Iterable

from backend.training.diagram_ir import (
    CanvasSpec,
    DiagramIRDocument,
    EdgeSpec,
    GroupSpec,
    LayoutSpec,
    NodeSpec,
    NodeStyle,
    SizeHint,
)


@dataclass(slots=True)
class CanonicalizationReport:
    source_nodes: int
    source_edges: int
    canonical_nodes: int
    canonical_edges: int
    text_nodes_aggregated: int
    shape_nodes_aggregated: int
    target_chars: int

    def to_dict(self) -> dict[str, int]:
        return {
            "source_nodes": self.source_nodes,
            "source_edges": self.source_edges,
            "canonical_nodes": self.canonical_nodes,
            "canonical_edges": self.canonical_edges,
            "text_nodes_aggregated": self.text_nodes_aggregated,
            "shape_nodes_aggregated": self.shape_nodes_aggregated,
            "target_chars": self.target_chars,
        }


_SPACE_RE = re.compile(r"\s+")


def compact_ir_json(document: DiagramIRDocument | dict[str, Any]) -> str:
    payload = document.to_dict() if isinstance(document, DiagramIRDocument) else document
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _clean_label(label: str, max_chars: int) -> str:
    cleaned = _SPACE_RE.sub(" ", label or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 3)].rstrip() + "..."


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x, y, width, height)):
        return None
    if width <= 0 or height <= 0:
        return None
    return (x, y, width, height)


def _node_bbox(node: NodeSpec) -> tuple[float, float, float, float] | None:
    return _bbox(node.metadata.get("bbox"))


def _union_bbox(boxes: Iterable[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    boxes = list(boxes)
    if not boxes:
        return None
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return (left, top, right - left, bottom - top)


def _size_hint(bbox: tuple[float, float, float, float] | None) -> SizeHint:
    if bbox is None:
        return SizeHint(min_width=80, min_height=32)
    return SizeHint(
        min_width=max(16, int(round(bbox[2]))),
        min_height=max(12, int(round(bbox[3]))),
    )


def _canonical_node(
    node_id: str,
    kind: str,
    label: str,
    bbox: tuple[float, float, float, float] | None,
    *,
    source_ids: list[str],
    style: NodeStyle | None = None,
) -> NodeSpec:
    metadata: dict[str, Any] = {
        "source_ids": source_ids,
        "canonical": True,
        "source_count": len(source_ids),
    }
    if bbox is not None:
        metadata["bbox"] = [round(value, 2) for value in bbox]
    return NodeSpec(
        id=node_id,
        kind=kind or "box",
        label=_clean_label(label or node_id, 120),
        size_hint=_size_hint(bbox),
        style=style or NodeStyle(),
        metadata=metadata,
    )


def _group_rows(nodes: list[NodeSpec], max_blocks: int) -> list[list[NodeSpec]]:
    if not nodes:
        return []
    ordered = sorted(nodes, key=lambda node: ((_node_bbox(node) or (0, 0, 0, 0))[1], (_node_bbox(node) or (0, 0, 0, 0))[0]))
    rows: list[list[NodeSpec]] = []
    for node in ordered:
        bbox = _node_bbox(node)
        if bbox is None:
            rows.append([node])
            continue
        y_mid = bbox[1] + bbox[3] / 2
        if rows:
            prev_boxes = [_node_bbox(item) for item in rows[-1]]
            prev_boxes = [box for box in prev_boxes if box is not None]
            prev = _union_bbox(prev_boxes)
            if prev is not None and abs(y_mid - (prev[1] + prev[3] / 2)) <= max(10.0, prev[3] * 0.75):
                rows[-1].append(node)
                continue
        rows.append([node])

    if len(rows) <= max_blocks:
        return rows

    chunk_size = max(1, math.ceil(len(rows) / max_blocks))
    blocks: list[list[NodeSpec]] = []
    for index in range(0, len(rows), chunk_size):
        block: list[NodeSpec] = []
        for row in rows[index : index + chunk_size]:
            block.extend(row)
        blocks.append(block)
    return blocks


def _aggregate_nodes(
    nodes: list[NodeSpec],
    *,
    prefix: str,
    kind: str,
    max_blocks: int,
) -> tuple[list[NodeSpec], int]:
    if len(nodes) <= max_blocks:
        canonical = []
        for index, node in enumerate(nodes, start=1):
            bbox = _node_bbox(node)
            canonical.append(
                _canonical_node(
                    f"{prefix}{index:03d}",
                    kind if kind != "shape" else node.kind,
                    node.label,
                    bbox,
                    source_ids=[node.id],
                    style=node.style,
                )
            )
        return canonical, 0

    blocks = _group_rows(nodes, max_blocks)
    canonical: list[NodeSpec] = []
    aggregated = 0
    for index, block in enumerate(blocks, start=1):
        boxes = [_node_bbox(node) for node in block]
        bbox = _union_bbox(box for box in boxes if box is not None)
        labels = [_clean_label(node.label, 40) for node in sorted(block, key=lambda node: ((_node_bbox(node) or (0, 0, 0, 0))[0]))]
        source_ids = [node.id for node in block]
        aggregated += max(0, len(block) - 1)
        canonical.append(
            _canonical_node(
                f"{prefix}{index:03d}",
                "text_block" if kind == "text" else "shape_cluster",
                " | ".join(label for label in labels if label),
                bbox,
                source_ids=source_ids,
                style=block[0].style if block else None,
            )
        )
    return canonical, aggregated


def _canonical_edges(
    source_edges: list[EdgeSpec],
    id_map: dict[str, str],
    max_edges: int,
) -> list[EdgeSpec]:
    edges: list[EdgeSpec] = []
    seen: set[tuple[str, str]] = set()
    for edge in source_edges:
        source = id_map.get(edge.source)
        target = id_map.get(edge.target)
        if source is None or target is None or source == target:
            continue
        pair = (source, target)
        if pair in seen:
            continue
        seen.add(pair)
        edges.append(
            EdgeSpec(
                id=f"e{len(edges) + 1:03d}",
                source=source,
                target=target,
                kind=edge.kind or "link",
                directed=edge.directed,
                label=_clean_label(edge.label, 60),
                routing=edge.routing,
                label_position=edge.label_position,
                style=edge.style,
            )
        )
        if len(edges) >= max_edges:
            break
    return edges


def canonicalize_diagram_ir(
    diagram_ir: dict[str, Any] | DiagramIRDocument,
    *,
    max_nodes: int = 160,
    max_edges: int = 240,
) -> tuple[DiagramIRDocument, CanonicalizationReport]:
    source = diagram_ir if isinstance(diagram_ir, DiagramIRDocument) else DiagramIRDocument.from_dict(diagram_ir)

    text_nodes = [node for node in source.nodes if node.kind == "text" or node.id.startswith("text-")]
    shape_nodes = [node for node in source.nodes if node not in text_nodes]
    shape_budget = min(len(shape_nodes), max(16, int(max_nodes * 0.55)))
    text_budget = max(8, max_nodes - shape_budget)

    canonical_shapes, shape_aggregated = _aggregate_nodes(
        shape_nodes,
        prefix="s",
        kind="shape",
        max_blocks=shape_budget,
    )
    canonical_text, text_aggregated = _aggregate_nodes(
        text_nodes,
        prefix="t",
        kind="text",
        max_blocks=text_budget,
    )
    nodes = canonical_shapes + canonical_text

    id_map: dict[str, str] = {}
    for canonical in nodes:
        for source_id in canonical.metadata.get("source_ids", []):
            id_map[source_id] = canonical.id

    edges = _canonical_edges(source.edges, id_map, max_edges)
    for node in nodes:
        node.metadata.pop("source_ids", None)
    groups: list[GroupSpec] = []
    if nodes:
        groups.append(
            GroupSpec(
                id="g001",
                members=[node.id for node in nodes],
                label=_clean_label(source.title or "canonical diagram", 80),
                metadata={"canonical": True},
            )
        )

    document = DiagramIRDocument(
        schema_version=source.schema_version,
        diagram_type=source.diagram_type,
        title=_clean_label(source.title or "canonical diagram", 80),
        canvas=CanvasSpec(
            width=source.canvas.width,
            height=source.canvas.height,
            grid=source.canvas.grid,
            origin_x=source.canvas.origin_x,
            origin_y=source.canvas.origin_y,
            padding=source.canvas.padding,
            background=source.canvas.background,
        ),
        nodes=nodes,
        edges=edges,
        groups=groups,
        layout=LayoutSpec(
            direction="freeform",
            alignment=source.layout.alignment,
            spacing=source.layout.spacing,
            node_spacing=source.layout.node_spacing,
            rank_spacing=source.layout.rank_spacing,
            snap_grid=source.layout.snap_grid,
            text_wrap=source.layout.text_wrap,
            edge_routing=source.layout.edge_routing,
        ),
        style=source.style,
        metadata={
            **source.metadata,
            "canonical": True,
            "source_node_count": len(source.nodes),
            "source_edge_count": len(source.edges),
        },
    )
    document.validate()
    target_chars = len(compact_ir_json(document))
    report = CanonicalizationReport(
        source_nodes=len(source.nodes),
        source_edges=len(source.edges),
        canonical_nodes=len(document.nodes),
        canonical_edges=len(document.edges),
        text_nodes_aggregated=text_aggregated,
        shape_nodes_aggregated=shape_aggregated,
        target_chars=target_chars,
    )
    return document, report
