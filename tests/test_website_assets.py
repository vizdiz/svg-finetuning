from __future__ import annotations

from pathlib import Path


def test_vite_app_is_static_preview_board():
    root = Path(__file__).resolve().parents[1]
    app = (root / "src" / "App.jsx").read_text()
    styles = (root / "src" / "App.css").read_text()
    index_css = (root / "index.css").read_text()

    assert "preview-grid" in app
    assert "PROMPT" in app
    assert "DRAFT & FEEDBACK" in app
    assert "REVISE & DOWNLOAD" in app
    assert "coming soon." in app
    assert "what should change?" in app
    assert "revise with feedback" in app
    assert "download draft" in app
    assert "/api/" not in app
    assert "static-copy" in app
    assert "family=IBM+Plex+Mono" in index_css
    assert "border-radius" not in styles
    assert "preview-grid" in styles
    assert "font-size: 56px" in styles
    assert "font-size: 30px" in styles
    assert "font-size: 10px" in styles
