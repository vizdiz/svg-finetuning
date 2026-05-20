"""
training.diagram_ir

Versioned, compiler-friendly intermediate representation for technical
diagrams.

This schema is intentionally narrow:
  - it models diagram structure, not raw SVG syntax
  - geometry stays mostly in the compiler
  - the model predicts declarative nodes, edges, groups, and layout hints

The schema is designed to be:
  - JSON serializable
  - round-trip stable
  - easy to validate deterministically
  - strict enough to catch malformed structure early

The compiler can later turn this into exact SVG geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

SCHEMA_VERSION = "0.1"

_LAYOUT_DIRECTIONS = {"horizontal", "vertical", "freeform"}
_LAYOUT_ALIGNMENTS = {"start", "center", "end", "stretch"}
_EDGE_ROUTINGS = {"auto", "orthogonal", "straight", "curved"}
_LABEL_POSITIONS = {"start", "mid", "end", "inside", "outside"}
_PORT_SIDES = {"top", "right", "bottom", "left", "auto"}
_TEXT_ALIGNMENTS = {"left", "center", "right"}


class IRSchemaError(ValueError):
    """Raised when an IR document fails deterministic validation."""


@dataclass(slots=True)
class CanvasSpec:
    width: int = 1200
    height: int = 800
    grid: int = 8
    origin_x: int = 0
    origin_y: int = 0
    padding: int = 32
    background: str = "#F0EFEB"

    def validate(self) -> None:
        if self.width <= 0:
            raise IRSchemaError(f"canvas.width must be > 0, got {self.width}")
        if self.height <= 0:
            raise IRSchemaError(f"canvas.height must be > 0, got {self.height}")
        if self.grid <= 0:
            raise IRSchemaError(f"canvas.grid must be > 0, got {self.grid}")
        if self.padding < 0:
            raise IRSchemaError(f"canvas.padding must be >= 0, got {self.padding}")


@dataclass(slots=True)
class SizeHint:
    min_width: int | None = None
    min_height: int | None = None
    aspect_ratio: float | None = None

    def validate(self) -> None:
        if self.min_width is not None and self.min_width <= 0:
            raise IRSchemaError(
                f"size_hint.min_width must be > 0 when set, got {self.min_width}"
            )
        if self.min_height is not None and self.min_height <= 0:
            raise IRSchemaError(
                f"size_hint.min_height must be > 0 when set, got {self.min_height}"
            )
        if self.aspect_ratio is not None and self.aspect_ratio <= 0:
            raise IRSchemaError(
                f"size_hint.aspect_ratio must be > 0 when set, got {self.aspect_ratio}"
            )


@dataclass(slots=True)
class TextStyle:
    font_size: int | None = None
    weight: int | None = None
    italic: bool = False
    align: str = "center"
    color: str | None = None

    def validate(self) -> None:
        if self.font_size is not None and self.font_size <= 0:
            raise IRSchemaError(
                f"text_style.font_size must be > 0 when set, got {self.font_size}"
            )
        if self.weight is not None and self.weight <= 0:
            raise IRSchemaError(f"text_style.weight must be > 0 when set, got {self.weight}")
        if self.align not in _TEXT_ALIGNMENTS:
            raise IRSchemaError(
                f"text_style.align must be one of {_TEXT_ALIGNMENTS}, got {self.align!r}"
            )


@dataclass(slots=True)
class NodeStyle:
    fill: str | None = None
    stroke: str | None = None
    stroke_width: float | None = None
    text_color: str | None = None
    opacity: float | None = None
    text: TextStyle = field(default_factory=TextStyle)

    def validate(self) -> None:
        if self.stroke_width is not None and self.stroke_width <= 0:
            raise IRSchemaError(
                f"node_style.stroke_width must be > 0 when set, got {self.stroke_width}"
            )
        if self.opacity is not None and not (0 <= self.opacity <= 1):
            raise IRSchemaError(
                f"node_style.opacity must be in [0, 1] when set, got {self.opacity}"
            )
        self.text.validate()


@dataclass(slots=True)
class PortSpec:
    name: str
    side: str = "auto"
    offset: float | None = None
    label: str | None = None

    def validate(self) -> None:
        if not self.name:
            raise IRSchemaError("port.name must be non-empty")
        if self.side not in _PORT_SIDES:
            raise IRSchemaError(f"port.side must be one of {_PORT_SIDES}, got {self.side!r}")
        if self.offset is not None and not (0 <= self.offset <= 1):
            raise IRSchemaError(
                f"port.offset must be in [0, 1] when set, got {self.offset}"
            )


@dataclass(slots=True)
class NodeSpec:
    id: str
    kind: str
    label: str = ""
    role: str | None = None
    group_id: str | None = None
    size_hint: SizeHint = field(default_factory=SizeHint)
    style: NodeStyle = field(default_factory=NodeStyle)
    ports: list[PortSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id:
            raise IRSchemaError("node.id must be non-empty")
        if not self.kind:
            raise IRSchemaError("node.kind must be non-empty")
        self.size_hint.validate()
        self.style.validate()
        seen_ports: set[str] = set()
        for port in self.ports:
            port.validate()
            if port.name in seen_ports:
                raise IRSchemaError(f"node {self.id!r} has duplicate port {port.name!r}")
            seen_ports.add(port.name)


@dataclass(slots=True)
class EdgeStyle:
    stroke: str | None = None
    stroke_width: float | None = None
    dasharray: str | None = None
    arrowhead: str | None = "arrow"

    def validate(self) -> None:
        if self.stroke_width is not None and self.stroke_width <= 0:
            raise IRSchemaError(
                f"edge_style.stroke_width must be > 0 when set, got {self.stroke_width}"
            )


@dataclass(slots=True)
class EdgeSpec:
    id: str
    source: str
    target: str
    kind: str = "link"
    directed: bool = True
    label: str = ""
    routing: str = "auto"
    label_position: str = "mid"
    source_port: str | None = None
    target_port: str | None = None
    style: EdgeStyle = field(default_factory=EdgeStyle)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id:
            raise IRSchemaError("edge.id must be non-empty")
        if not self.source:
            raise IRSchemaError("edge.source must be non-empty")
        if not self.target:
            raise IRSchemaError("edge.target must be non-empty")
        if self.routing not in _EDGE_ROUTINGS:
            raise IRSchemaError(
                f"edge.routing must be one of {_EDGE_ROUTINGS}, got {self.routing!r}"
            )
        if self.label_position not in _LABEL_POSITIONS:
            raise IRSchemaError(
                f"edge.label_position must be one of {_LABEL_POSITIONS}, got {self.label_position!r}"
            )
        self.style.validate()


@dataclass(slots=True)
class GroupStyle:
    fill: str | None = None
    stroke: str | None = None
    stroke_width: float | None = None

    def validate(self) -> None:
        if self.stroke_width is not None and self.stroke_width <= 0:
            raise IRSchemaError(
                f"group_style.stroke_width must be > 0 when set, got {self.stroke_width}"
            )


@dataclass(slots=True)
class GroupSpec:
    id: str
    members: list[str]
    kind: str = "cluster"
    label: str = ""
    direction: str = "vertical"
    padding: int = 24
    style: GroupStyle = field(default_factory=GroupStyle)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id:
            raise IRSchemaError("group.id must be non-empty")
        if not self.members:
            raise IRSchemaError(f"group {self.id!r} must contain at least one member")
        if self.direction not in _LAYOUT_DIRECTIONS:
            raise IRSchemaError(
                f"group.direction must be one of {_LAYOUT_DIRECTIONS}, got {self.direction!r}"
            )
        if self.padding < 0:
            raise IRSchemaError(f"group.padding must be >= 0, got {self.padding}")
        self.style.validate()


@dataclass(slots=True)
class LayoutSpec:
    direction: str = "horizontal"
    alignment: str = "center"
    spacing: int = 48
    node_spacing: int | None = None
    rank_spacing: int | None = None
    snap_grid: int = 8
    text_wrap: str = "balanced"
    edge_routing: str = "auto"

    def validate(self) -> None:
        if self.direction not in _LAYOUT_DIRECTIONS:
            raise IRSchemaError(
                f"layout.direction must be one of {_LAYOUT_DIRECTIONS}, got {self.direction!r}"
            )
        if self.alignment not in _LAYOUT_ALIGNMENTS:
            raise IRSchemaError(
                f"layout.alignment must be one of {_LAYOUT_ALIGNMENTS}, got {self.alignment!r}"
            )
        if self.spacing <= 0:
            raise IRSchemaError(f"layout.spacing must be > 0, got {self.spacing}")
        if self.node_spacing is not None and self.node_spacing <= 0:
            raise IRSchemaError(
                f"layout.node_spacing must be > 0 when set, got {self.node_spacing}"
            )
        if self.rank_spacing is not None and self.rank_spacing <= 0:
            raise IRSchemaError(
                f"layout.rank_spacing must be > 0 when set, got {self.rank_spacing}"
            )
        if self.snap_grid <= 0:
            raise IRSchemaError(f"layout.snap_grid must be > 0, got {self.snap_grid}")
        if self.edge_routing not in _EDGE_ROUTINGS:
            raise IRSchemaError(
                f"layout.edge_routing must be one of {_EDGE_ROUTINGS}, got {self.edge_routing!r}"
            )


@dataclass(slots=True)
class DocumentStyle:
    theme: str = "technical"
    font_family: str = "IBM Plex Mono"
    background: str | None = None
    foreground: str | None = None
    line_color: str | None = None
    accent_color: str | None = None


@dataclass(slots=True)
class DiagramIRDocument:
    schema_version: str = SCHEMA_VERSION
    diagram_type: str = "flowchart"
    title: str = ""
    canvas: CanvasSpec = field(default_factory=CanvasSpec)
    nodes: list[NodeSpec] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)
    groups: list[GroupSpec] = field(default_factory=list)
    layout: LayoutSpec = field(default_factory=LayoutSpec)
    style: DocumentStyle = field(default_factory=DocumentStyle)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.schema_version:
            raise IRSchemaError("schema_version must be non-empty")
        if self.diagram_type is None or not str(self.diagram_type).strip():
            raise IRSchemaError("diagram_type must be non-empty")

        self.canvas.validate()
        self.layout.validate()

        node_ids: list[str] = []
        for node in self.nodes:
            node.validate()
            node_ids.append(node.id)
        if len(node_ids) != len(set(node_ids)):
            raise IRSchemaError("node ids must be unique")

        node_id_set = set(node_ids)

        edge_ids: list[str] = []
        for edge in self.edges:
            edge.validate()
            edge_ids.append(edge.id)
            if edge.source not in node_id_set:
                raise IRSchemaError(
                    f"edge {edge.id!r} references unknown source node {edge.source!r}"
                )
            if edge.target not in node_id_set:
                raise IRSchemaError(
                    f"edge {edge.id!r} references unknown target node {edge.target!r}"
                )
            if edge.source_port is not None:
                self._validate_port_reference(edge.source, edge.source_port)
            if edge.target_port is not None:
                self._validate_port_reference(edge.target, edge.target_port)
        if len(edge_ids) != len(set(edge_ids)):
            raise IRSchemaError("edge ids must be unique")

        group_ids: list[str] = []
        for group in self.groups:
            group.validate()
            group_ids.append(group.id)
            for member in group.members:
                if member not in node_id_set:
                    raise IRSchemaError(
                        f"group {group.id!r} references unknown node {member!r}"
                    )
        if len(group_ids) != len(set(group_ids)):
            raise IRSchemaError("group ids must be unique")

    def _validate_port_reference(self, node_id: str, port_name: str) -> None:
        node = next((item for item in self.nodes if item.id == node_id), None)
        if node is None:
            raise IRSchemaError(f"unknown node {node_id!r}")
        if port_name not in {port.name for port in node.ports}:
            raise IRSchemaError(f"node {node_id!r} has no port named {port_name!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiagramIRDocument":
        def _canvas(value: dict[str, Any] | CanvasSpec | None) -> CanvasSpec:
            if isinstance(value, CanvasSpec):
                return value
            return CanvasSpec(**(value or {}))

        def _size_hint(value: dict[str, Any] | SizeHint | None) -> SizeHint:
            if isinstance(value, SizeHint):
                return value
            return SizeHint(**(value or {}))

        def _text_style(value: dict[str, Any] | TextStyle | None) -> TextStyle:
            if isinstance(value, TextStyle):
                return value
            return TextStyle(**(value or {}))

        def _node_style(value: dict[str, Any] | NodeStyle | None) -> NodeStyle:
            if isinstance(value, NodeStyle):
                return value
            payload = dict(value or {})
            payload["text"] = _text_style(payload.get("text"))
            return NodeStyle(**payload)

        def _ports(values: list[dict[str, Any]] | list[PortSpec] | None) -> list[PortSpec]:
            ports: list[PortSpec] = []
            for value in values or []:
                ports.append(value if isinstance(value, PortSpec) else PortSpec(**value))
            return ports

        def _edge_style(value: dict[str, Any] | EdgeStyle | None) -> EdgeStyle:
            if isinstance(value, EdgeStyle):
                return value
            return EdgeStyle(**(value or {}))

        def _group_style(value: dict[str, Any] | GroupStyle | None) -> GroupStyle:
            if isinstance(value, GroupStyle):
                return value
            return GroupStyle(**(value or {}))

        def _layout(value: dict[str, Any] | LayoutSpec | None) -> LayoutSpec:
            if isinstance(value, LayoutSpec):
                return value
            return LayoutSpec(**(value or {}))

        def _document_style(value: dict[str, Any] | DocumentStyle | None) -> DocumentStyle:
            if isinstance(value, DocumentStyle):
                return value
            return DocumentStyle(**(value or {}))

        nodes = []
        for raw_node in data.get("nodes", []) or []:
            payload = dict(raw_node)
            payload["size_hint"] = _size_hint(payload.get("size_hint"))
            payload["style"] = _node_style(payload.get("style"))
            payload["ports"] = _ports(payload.get("ports"))
            nodes.append(NodeSpec(**payload))

        edges = []
        for raw_edge in data.get("edges", []) or []:
            payload = dict(raw_edge)
            payload["style"] = _edge_style(payload.get("style"))
            edges.append(EdgeSpec(**payload))

        groups = []
        for raw_group in data.get("groups", []) or []:
            payload = dict(raw_group)
            payload["style"] = _group_style(payload.get("style"))
            groups.append(GroupSpec(**payload))

        doc = cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            diagram_type=data.get("diagram_type", "flowchart"),
            title=data.get("title", ""),
            canvas=_canvas(data.get("canvas")),
            nodes=nodes,
            edges=edges,
            groups=groups,
            layout=_layout(data.get("layout")),
            style=_document_style(data.get("style")),
            metadata=dict(data.get("metadata", {})),
        )
        doc.validate()
        return doc

    @classmethod
    def from_json(cls, raw: str) -> "DiagramIRDocument":
        return cls.from_dict(json.loads(raw))

    def node_map(self) -> dict[str, NodeSpec]:
        return {node.id: node for node in self.nodes}

