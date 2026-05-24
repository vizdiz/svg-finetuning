from __future__ import annotations

from dataclasses import dataclass, field

from lxml import etree


@dataclass
class RawSVG:
    svg_string: str
    source_url: str
    source_id: str
    domain: str
    metadata: dict = field(default_factory=dict)


def count_svg_elements(svg_string: str) -> int:
    root = etree.fromstring(svg_string.encode())
    return sum(1 for _ in root.iter()) - 1
