"""
dataset_pipeline.processing.validator

Deterministic SVG safety validation plus non-blocking quality flagging.

Hard rejects are reserved for inputs that are unsafe, non-portable, malformed,
or likely to destabilise parsing/rendering. Quality issues are warnings only;
they are intended for aggregate monitoring so useful but messy diagrams are not
discarded prematurely.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from lxml import etree

from backend.dataset_pipeline.config import ScrapeConfig

logger = logging.getLogger(__name__)

_SVG_NS = "http://www.w3.org/2000/svg"
_XMLNS_NS = "http://www.w3.org/2000/xmlns/"
_XLINK_NS = "http://www.w3.org/1999/xlink"

_PROCESSING_INSTRUCTION_RE = re.compile(r"<\?(?!xml\s)[\s\S]*?\?>", re.IGNORECASE)
_CSS_IMPORT_RE = re.compile(r"@import\b", re.IGNORECASE)
_CSS_URL_RE = re.compile(r"url\(\s*['\"]?([^'\"\)\s]+)", re.IGNORECASE)
_REMOTE_REF_RE = re.compile(r"^(?:https?:|file:|//)", re.IGNORECASE)
_BASE64ISH_RE = re.compile(r"^[A-Za-z0-9+/=\s]{4096,}$")
_NUMBER_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", re.IGNORECASE)

_EDITOR_NAMESPACE_MARKERS = (
    "inkscape",
    "sodipodi",
    "adobe",
    "illustrator",
    "sketch",
)

_QUALITY_WARNING_PENALTIES = {
    "size_under_limit": 10,
    "metadata_present": 2,
    "editor_metadata_present": 6,
    "style_attrs_present": 4,
    "unusual_aspect_ratio": 5,
    "too_simple": 12,
    "many_text_nodes": 6,
    "many_paths": 6,
    "large_defs": 4,
    "many_hidden_elements": 8,
}


@dataclass
class SVGValidationResult:
    ok: bool
    hard_reject_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def reason(self) -> str:
        if self.ok:
            return "ok"
        return self.hard_reject_reasons[0] if self.hard_reject_reasons else "rejected"


def _local_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    return etree.QName(name).localname.lower()


def _namespace(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    return etree.QName(name).namespace or ""


def _parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    match = _NUMBER_RE.match(value)
    if not match:
        return None
    try:
        number = float(match.group(1))
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _parse_viewbox(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    parts = re.split(r"[\s,]+", value.strip())
    if len(parts) != 4:
        return None
    try:
        nums = tuple(float(part) for part in parts)
    except ValueError:
        return None
    if not all(math.isfinite(n) for n in nums):
        return None
    return nums


def _is_external_or_local_reference(value: str) -> bool:
    stripped = value.strip()
    if not stripped or stripped.startswith("#"):
        return False
    if _REMOTE_REF_RE.match(stripped):
        return True
    if stripped.startswith("/") or stripped.startswith("\\"):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", stripped):
        return True
    return False


def _check_css_urls(css: str, hard_rejects: set[str]) -> None:
    if _CSS_IMPORT_RE.search(css):
        hard_rejects.add("css_import")
    for match in _CSS_URL_RE.finditer(css):
        ref = match.group(1).strip()
        if ref.lower().startswith("data:"):
            hard_rejects.add("embedded_data_reference")
        elif _is_external_or_local_reference(ref):
            hard_rejects.add("remote_url_function")


def _add_warning(warnings: set[str], condition: bool, reason: str) -> None:
    if condition:
        warnings.add(reason)


def svg_quality_score(result: SVGValidationResult) -> int:
    """
    Deterministic quality score for ranking usable SVGs before captioning.

    Higher is better. Hard rejections are not scored because they should not
    reach this stage.
    """

    score = 100
    for warning in result.warnings:
        if warning.startswith("too_simple"):
            score -= _QUALITY_WARNING_PENALTIES["too_simple"]
        else:
            score -= _QUALITY_WARNING_PENALTIES.get(warning, 1)

    nodes = int(result.stats.get("nodes", 0) or 0)
    paths = int(result.stats.get("paths", 0) or 0)
    texts = int(result.stats.get("texts", 0) or 0)
    max_attr_chars = int(result.stats.get("max_attr_chars", 0) or 0)

    if nodes > 1000:
        score -= min(20, (nodes - 1000) // 500)
    if paths > 500:
        score -= min(10, (paths - 500) // 250)
    if texts > 50:
        score -= min(10, (texts - 50) // 50)
    if max_attr_chars > 20_000:
        score -= min(10, (max_attr_chars - 20_000) // 20_000)

    return max(0, score)


def validate_svg_detailed(svg_string: str, config: ScrapeConfig | None = None) -> SVGValidationResult:
    if config is None:
        config = ScrapeConfig()

    hard_rejects: set[str] = set()
    warnings: set[str] = set()
    stats: dict[str, Any] = {"bytes": len(svg_string.encode())}

    if stats["bytes"] > config.max_svg_bytes:
        hard_rejects.add("size_over_limit")

    if _PROCESSING_INSTRUCTION_RE.search(svg_string):
        hard_rejects.add("processing_instruction")

    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        root = etree.fromstring(svg_string.encode(), parser)
    except etree.XMLSyntaxError as exc:
        return SVGValidationResult(
            ok=False,
            hard_reject_reasons=[f"xml_parse_fail: {exc}"],
            warnings=sorted(warnings),
            stats=stats,
        )

    if _local_name(root.tag) != "svg" or _namespace(root.tag) not in ("", _SVG_NS):
        hard_rejects.add("non_svg_root")

    viewbox = _parse_viewbox(root.get("viewBox"))
    width = _parse_number(root.get("width"))
    height = _parse_number(root.get("height"))
    stats.update(
        {
            "has_viewbox": viewbox is not None,
            "has_width_height": width is not None and height is not None,
            "width": width,
            "height": height,
        }
    )

    viewport_width = viewbox[2] if viewbox else width
    viewport_height = viewbox[3] if viewbox else height
    if viewport_width is not None and viewport_height is not None and viewport_height != 0:
        stats["aspect_ratio"] = viewport_width / viewport_height

    if viewbox is None and (width is None or height is None):
        hard_rejects.add("no_dimensions")
    elif viewport_width is None or viewport_height is None:
        hard_rejects.add("zero_or_negative_dimensions")
    elif viewport_width <= 0 or viewport_height <= 0:
        hard_rejects.add("zero_or_negative_dimensions")
    else:
        aspect_ratio = viewport_width / viewport_height
        if aspect_ratio < config.hard_min_aspect_ratio or aspect_ratio > config.hard_max_aspect_ratio:
            hard_rejects.add("absurd_aspect_ratio")
        _add_warning(
            warnings,
            aspect_ratio < config.warning_min_aspect_ratio or aspect_ratio > config.warning_max_aspect_ratio,
            "unusual_aspect_ratio",
        )

    node_count = 0
    path_count = 0
    text_count = 0
    style_attr_count = 0
    metadata_count = 0
    defs_count = 0
    total_path_chars = 0
    max_attr_chars = 0
    hidden_count = 0
    editor_attr_count = 0
    editor_ns_count = 0

    for elem in root.iter():
        node_count += 1
        tag = _local_name(elem.tag)
        ns = _namespace(elem.tag).lower()

        if tag == "script":
            hard_rejects.add("script_tag")
        elif tag == "foreignobject":
            hard_rejects.add("foreign_object")
        elif tag == "image":
            hard_rejects.add("raster_image")
        elif tag == "metadata":
            metadata_count += 1
        elif tag == "defs":
            defs_count += 1
        elif tag == "path":
            path_count += 1
            total_path_chars += len(elem.get("d", ""))
        elif tag == "text":
            text_count += 1
        elif tag == "style":
            _check_css_urls("".join(elem.itertext()), hard_rejects)

        if any(marker in ns for marker in _EDITOR_NAMESPACE_MARKERS):
            editor_ns_count += 1

        for attr_name, attr_value in elem.attrib.items():
            attr = _local_name(attr_name)
            attr_ns = _namespace(attr_name).lower()
            value = attr_value or ""
            max_attr_chars = max(max_attr_chars, len(value))

            if attr.startswith("on"):
                hard_rejects.add("event_handler_attr")
            if attr in ("href", "src") or (attr_ns == _XLINK_NS and attr == "href"):
                if value.strip().lower().startswith("data:"):
                    hard_rejects.add("embedded_data_reference")
                elif _is_external_or_local_reference(value):
                    hard_rejects.add("external_reference")
            if attr == "style":
                style_attr_count += 1
                _check_css_urls(value, hard_rejects)
            elif "url(" in value.lower():
                _check_css_urls(value, hard_rejects)
            if attr in ("display", "visibility", "opacity") and value.strip().lower() in ("none", "hidden", "0"):
                hidden_count += 1
            if len(value) > config.max_single_attr_chars:
                hard_rejects.add("single_attribute_over_limit")
            if _BASE64ISH_RE.match(value):
                hard_rejects.add("base64_blob_attribute")
            if attr_ns == _XMLNS_NS and any(marker in value.lower() for marker in _EDITOR_NAMESPACE_MARKERS):
                editor_ns_count += 1
            if any(marker in attr.lower() or marker in attr_ns for marker in _EDITOR_NAMESPACE_MARKERS):
                editor_attr_count += 1

    stats.update(
        {
            "nodes": node_count,
            "paths": path_count,
            "texts": text_count,
            "style_attrs": style_attr_count,
            "metadata": metadata_count,
            "defs": defs_count,
            "total_path_chars": total_path_chars,
            "max_attr_chars": max_attr_chars,
            "hidden_elements_or_attrs": hidden_count,
            "editor_attrs": editor_attr_count,
            "editor_namespaces": editor_ns_count,
        }
    )

    if node_count - 1 < config.min_svg_elements:
        warnings.add(f"too_simple: {node_count - 1} elements")
    if node_count > config.max_svg_nodes:
        hard_rejects.add("node_count_over_limit")
    if total_path_chars > config.max_total_path_chars:
        hard_rejects.add("path_data_over_limit")

    if stats["bytes"] < config.min_svg_bytes:
        warnings.add("size_under_limit")

    _add_warning(warnings, metadata_count > 0, "metadata_present")
    _add_warning(warnings, editor_ns_count > 0 or editor_attr_count > 0, "editor_metadata_present")
    _add_warning(warnings, style_attr_count > 0, "style_attrs_present")
    _add_warning(warnings, text_count > 300, "many_text_nodes")
    _add_warning(warnings, path_count > 5000, "many_paths")
    _add_warning(warnings, defs_count > 20, "large_defs")
    _add_warning(warnings, hidden_count > 100, "many_hidden_elements")

    return SVGValidationResult(
        ok=not hard_rejects,
        hard_reject_reasons=sorted(hard_rejects),
        warnings=sorted(warnings),
        stats=stats,
    )


def validate_svg(svg_string: str, config: ScrapeConfig = None) -> tuple[bool, str]:
    result = validate_svg_detailed(svg_string, config)
    if result.ok:
        return True, "ok"

    reasons = result.hard_reject_reasons
    if any(reason.startswith("xml_parse_fail") for reason in reasons):
        return False, next(reason for reason in reasons if reason.startswith("xml_parse_fail"))
    if "raster_image" in reasons:
        return False, "contains_raster"
    if "no_dimensions" in reasons:
        return False, "no_dimensions"
    if "size_over_limit" in reasons:
        return False, f"size_out_of_range: {result.stats.get('bytes', 0)}"
    return False, result.reason
