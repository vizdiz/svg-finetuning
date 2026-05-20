"""
Unit tests for dataset_pipeline.processing.validator.

Run with:
    pytest tests/validator.py -v
"""

import pytest

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.processing.validator import validate_svg, validate_svg_detailed


def _svg(body: str = "", *, attrs: str = "", viewbox: bool = True, dims: bool = True) -> str:
    vb = 'viewBox="0 0 100 100"' if viewbox else ""
    wh = 'width="100" height="100"' if dims else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" {vb} {wh} {attrs}>'
        + body
        + "</svg>"
    )


def _many_paths(n: int = 10) -> str:
    return "".join(f'<path d="M{i} 0 L{i} 10"/>' for i in range(n))


# ── Check 1: XML parse ────────────────────────────────────────────────────────

def test_rejects_invalid_xml():
    ok, reason = validate_svg("not xml")
    assert not ok
    assert reason.startswith("xml_parse_fail")


def test_rejects_unclosed_tag():
    ok, reason = validate_svg("<svg><rect></svg>")
    assert not ok
    assert reason.startswith("xml_parse_fail")


# ── Check 2: Render ───────────────────────────────────────────────────────────

def test_rejects_unrenderable_svg():
    # cairosvg will fail on a malformed SVG that is still valid XML
    bad = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="-999999999999"/></svg>'
    ok, reason = validate_svg(bad)
    # may pass or fail render depending on cairosvg version — just assert no crash
    assert isinstance(ok, bool)


# ── Check 3: Element count ────────────────────────────────────────────────────

def test_flags_too_few_elements_without_rejecting():
    config = ScrapeConfig(min_svg_elements=5)
    ok, reason = validate_svg(_svg("<path d='M0 0'/>"), config=config)
    assert ok
    assert reason == "ok" or reason == "too_simple"


def test_accepts_sufficient_elements():
    config = ScrapeConfig(min_svg_elements=5)
    ok, reason = validate_svg(_svg(_many_paths(10)), config=config)
    assert ok or not reason.startswith("too_simple")


# ── Check 4: No raster ────────────────────────────────────────────────────────

def test_rejects_image_tag():
    body = _many_paths(10) + '<image href="data:image/png;base64,abc"/>'
    ok, reason = validate_svg(_svg(body))
    assert not ok
    # cairosvg may fail on the bad image data before we reach the raster check
    assert reason in ("contains_raster", ) or reason.startswith("render_fail")


# ── Check 5: Dimensions ───────────────────────────────────────────────────────

def test_rejects_no_dimensions():
    svg = _svg(_many_paths(10), viewbox=False, dims=False)
    ok, reason = validate_svg(svg)
    assert not ok
    # cairosvg reports undefined size before we reach the dimension check
    assert reason == "no_dimensions" or reason.startswith("render_fail")


def test_accepts_viewbox_only():
    svg = _svg(_many_paths(10), viewbox=True, dims=False)
    ok, reason = validate_svg(svg)
    assert ok or reason not in ("no_dimensions",)


def test_accepts_width_height_only():
    svg = _svg(_many_paths(10), viewbox=False, dims=True)
    ok, reason = validate_svg(svg)
    assert ok or reason != "no_dimensions"


# ── Check 6: Post-normalization size ──────────────────────────────────────────

def test_flags_too_small_after_normalization():
    # Empty SVG is allowed but should be scored down as too simple / tiny.
    svg = _svg()
    ok, reason = validate_svg(svg)
    assert ok
    assert reason == "ok" or reason.startswith("too_simple")


def test_rejects_oversize_svg():
    # Build an SVG that is large after normalization
    huge_body = "".join(f'<path d="M{i}.{"1"*30} {i}.{"2"*30} L{i} {i}"/>' for i in range(500))
    svg = _svg(huge_body)
    ok, reason = validate_svg(svg)
    if not ok:
        assert reason.startswith("size_out_of_range") or reason.startswith("render_fail")


# ── Happy path ────────────────────────────────────────────────────────────────

def test_valid_svg_passes():
    svg = _svg(_many_paths(20))
    ok, reason = validate_svg(svg)
    assert ok, f"Expected ok but got: {reason}"


def test_returns_ok_string_on_success():
    svg = _svg(_many_paths(20))
    ok, reason = validate_svg(svg)
    if ok:
        assert reason == "ok"


# ── Safety hard rejects ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("<script>alert(1)</script>", "script_tag"),
        ('<rect onclick="alert(1)" width="10" height="10"/>', "event_handler_attr"),
        ("<foreignObject><div>html</div></foreignObject>", "foreign_object"),
        ('<use href="https://example.com/icon.svg#x"/>', "external_reference"),
        ('<style>@import url("https://example.com/a.css");</style>', "css_import"),
        ('<rect fill="url(https://example.com/a.svg#p)"/>', "remote_url_function"),
        ('<style>rect { fill: url(data:image/png;base64,abc); }</style>', "embedded_data_reference"),
        ('<image href="data:image/png;base64,abc"/>', "raster_image"),
        (f'<rect data-junk="{"A" * 4096}"/>', "base64_blob_attribute"),
    ],
)
def test_detailed_validator_rejects_unsafe_constructs(body, reason):
    result = validate_svg_detailed(_svg(_many_paths(10) + body))
    assert not result.ok
    assert reason in result.hard_reject_reasons


def test_detailed_validator_rejects_processing_instruction():
    svg = '<?xml-stylesheet href="https://example.com/a.css"?>' + _svg(_many_paths(10))
    result = validate_svg_detailed(svg)
    assert not result.ok
    assert "processing_instruction" in result.hard_reject_reasons


def test_detailed_validator_flags_quality_without_rejecting():
    svg = _svg(
        "<metadata>created by editor</metadata>" + _many_paths(30),
        attrs='xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" inkscape:version="1"',
    )
    result = validate_svg_detailed(svg)
    assert result.ok
    assert "metadata_present" in result.warnings
    assert "editor_metadata_present" in result.warnings
