from __future__ import annotations

from pathlib import Path


def test_vite_app_is_functional_preview_workspace():
    root = Path(__file__).resolve().parents[1]
    app = (root / "src" / "App.jsx").read_text()
    styles = (root / "src" / "App.css").read_text()
    index_css = (root / "index.css").read_text()

    assert "useState('prompt')" in app
    assert "execute-api.us-east-1.amazonaws.com/api/generate" in app
    assert "reference_images" in app
    assert "media_metadata" in app
    assert "attach media" in app
    assert "↓ download .svg" in app
    assert "what should change?" in app
    assert "iterate →" in app
    assert "unlock download →" in app
    assert "rating and feedback are required." in app
    assert "generated." in app
    assert "generation did not return svg." in app
    assert "skip" not in app
    assert "what was wrong? (optional)" not in app
    assert "/api/generate" in app
    assert "family=IBM+Plex+Mono" in index_css
    assert "border-radius" not in styles
    assert "box-shadow" not in styles
    assert "#FFFFFF" not in styles
    assert "font-size: 16px" in styles
    assert "font-size: 12px" in styles
    assert "font-size: 11px" in styles
    assert "font-size: 18px" in styles
