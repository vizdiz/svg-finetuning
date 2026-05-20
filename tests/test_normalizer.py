"""
Unit tests for dataset_pipeline.processing.normalizer.

Run with:
    pytest tests/normalizer.py -v
"""

import pytest
from lxml import etree

from backend.dataset_pipeline.processing.normalizer import normalize_svg


def _parse(svg: str) -> etree._Element:
    return etree.fromstring(normalize_svg(svg).encode())


def _svg(*inner: str, extra_attrs: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="200" height="100" viewBox="0 0 200 100" {extra_attrs}>'
        + "".join(inner)
        + "</svg>"
    )


# ── Comments ──────────────────────────────────────────────────────────────────

def test_removes_comments():
    out = normalize_svg(_svg("<!-- ignored -->", "<rect x='1' y='1' width='10' height='10'/>"))
    assert "<!--" not in out


# ── No-op transforms ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("t", [
    "translate(0,0)", "translate(0 0)", "translate(0)",
    "scale(1)", "scale(1,1)", "rotate(0)", "matrix(1,0,0,1,0,0)",
])
def test_removes_noop_transform(t):
    out = normalize_svg(_svg(f'<g transform="{t}"><rect x="1" y="1" width="5" height="5"/></g>'))
    assert 'transform=' not in out


def test_keeps_non_noop_transform():
    out = normalize_svg(_svg('<g transform="rotate(45)"><rect x="1" y="1" width="5" height="5"/></g>'))
    assert 'transform="rotate(45)"' in out


# ── Coordinate rounding ───────────────────────────────────────────────────────

@pytest.mark.parametrize("attr,value,expected", [
    ("cx", "50.123456", "50.1"),
    ("cy", "50.987654", "51.0"),
    ("r",  "40.111111", "40.1"),
    ("x",  "10.05",     "10.1"),
    ("y",  "0.0",       "0.0"),
    ("x1", "3.333",     "3.3"),
    ("y2", "99.99",     "100.0"),
    ("rx", "5.55",      "5.5"),   # 5.55 < 5.55 in IEEE 754 → rounds down
    ("ry", "5.44",      "5.4"),
    ("dx", "1.15",      "1.1"),  # 1.15 < 1.15 in IEEE 754 → rounds down
    ("dy", "2.25",      "2.2"),  # banker's rounding via Python round()
])
def test_rounds_coord_attr(attr, value, expected):
    tag = "circle" if attr in ("cx", "cy", "r") else "rect"
    extra = 'width="10" height="10"' if tag == "rect" else ""
    svg = _svg(f'<{tag} {attr}="{value}" {extra}/>')
    root = _parse(svg)
    elem = root.find(f".//{{{  'http://www.w3.org/2000/svg'}}}{tag}")
    assert elem.get(attr) == expected


def test_skips_percentage_value():
    out = normalize_svg(_svg('<rect x="50%" y="10" width="10" height="10"/>'))
    assert 'x="50%"' in out


def test_skips_value_with_units():
    out = normalize_svg(_svg('<rect x="10px" y="10" width="10" height="10"/>'))
    assert 'x="10px"' in out


def test_root_width_height_not_rounded():
    svg = _svg()  # root has width="200" height="100"
    root = _parse(svg)
    assert root.get("width") == "200"
    assert root.get("height") == "100"


def test_child_width_height_are_rounded():
    svg = _svg('<rect x="0" y="0" width="10.555" height="20.444"/>')
    root = _parse(svg)
    ns = "http://www.w3.org/2000/svg"
    rect = root.find(f"{{{ns}}}rect")
    assert rect.get("width") == "10.6"
    assert rect.get("height") == "20.4"


# ── Empty <g> removal ─────────────────────────────────────────────────────────

def test_removes_empty_g():
    out = normalize_svg(_svg("<g></g>", "<rect x='1' y='1' width='5' height='5'/>"))
    assert "<g/>" not in out and "<g>" not in out


def test_removes_nested_empty_g():
    out = normalize_svg(_svg("<g><g></g></g>", "<rect x='1' y='1' width='5' height='5'/>"))
    assert "<g>" not in out


def test_keeps_g_with_children():
    out = normalize_svg(_svg('<g><rect x="1" y="1" width="5" height="5"/></g>'))
    assert "<g>" in out


def test_keeps_g_with_text():
    out = normalize_svg(_svg('<g>some text</g>'))
    ns = "http://www.w3.org/2000/svg"
    root = etree.fromstring(out.encode())
    g = root.find(f"{{{ns}}}g")
    assert g is not None


# ── Output format ─────────────────────────────────────────────────────────────

def test_output_is_single_line():
    out = normalize_svg(_svg('<rect x="1" y="1" width="5" height="5"/>'))
    assert "\n" not in out


def test_no_xml_declaration():
    out = normalize_svg(_svg())
    assert not out.startswith("<?xml")


# ── Fault tolerance ───────────────────────────────────────────────────────────

def test_returns_original_on_invalid_xml():
    bad = "not xml at all"
    assert normalize_svg(bad) == bad
