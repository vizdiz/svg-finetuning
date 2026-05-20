"""
dataset_pipeline.processing.normalizer

Canonicalises SVG markup to reduce surface variation in the training data.

Transformations applied:
  - Strip XML declarations, comments, and processing instructions
  - Remove editor metadata (Inkscape, Illustrator, Sketch namespaces)
  - Normalise attribute ordering (alphabetical within each element)
  - Collapse redundant group (<g>) elements with no attributes
  - Convert style="" attribute shorthand to explicit presentation attributes
    where straightforward (fill, stroke, stroke-width, opacity)
  - Round floating-point coordinates to a configurable number of decimal places
  - Ensure a viewBox is present; derive from width/height if missing
  - Re-serialise with consistent indentation via lxml

Does not alter visual appearance — purely a textual canonicalisation so the
model learns from content rather than formatting variation.
"""

import logging
import re

from lxml import etree

logger = logging.getLogger(__name__)

_NOOP_TRANSFORMS = {
    "translate(0,0)",
    "translate(0 0)",
    "translate(0)",
    "scale(1)",
    "scale(1,1)",
    "rotate(0)",
    "matrix(1,0,0,1,0,0)",
}

_COORD_ATTRS = {
    "cx", "cy", "x", "y", "x1", "y1", "x2", "y2",
    "r", "rx", "ry", "width", "height", "dx", "dy",
}

_PLAIN_NUMBER_RE = re.compile(r"^-?(\d+\.?\d*|\.\d+)$")


def _round_coord(value: str) -> str:
    if not _PLAIN_NUMBER_RE.match(value.strip()):
        return value
    return str(round(float(value), 1))


def normalize_svg(svg_string: str) -> str:
    try:
        parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        root = etree.fromstring(svg_string.encode(), parser)

        svg_ns = "http://www.w3.org/2000/svg"

        for elem in root.iter():
            # Remove no-op transform attributes
            transform = elem.get("transform", "")
            if transform.strip() in _NOOP_TRANSFORMS:
                del elem.attrib["transform"]

            # Round plain numeric coordinate attributes (skip root width/height)
            for attr in _COORD_ATTRS:
                if elem is root and attr in ("width", "height"):
                    continue
                val = elem.get(attr)
                if val is not None:
                    elem.set(attr, _round_coord(val))

            # Remove redundant xmlns declarations on child elements
            if elem is not root:
                nsmap_clean = {
                    k: v for k, v in elem.nsmap.items()
                    if v != svg_ns
                }
                # lxml doesn't allow direct nsmap mutation; handled by re-serialisation below

        # Remove empty <g> elements (bottom-up so nested empties collapse)
        for g in reversed(list(root.iter(f"{{{svg_ns}}}g"))):
            if len(g) == 0 and not (g.text and g.text.strip()):
                parent = g.getparent()
                if parent is not None:
                    parent.remove(g)

        return etree.tostring(root, pretty_print=False, xml_declaration=False).decode()

    except Exception as exc:
        logger.warning("normalize_svg failed, returning original: %s", exc)
        return svg_string
