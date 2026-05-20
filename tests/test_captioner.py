"""
Unit tests for dataset_pipeline.processing.captioner.

Run with:
    pytest tests/captioner.py -v
"""

import sys
import types
import time
from types import SimpleNamespace

import pytest

from backend.dataset_pipeline.processing import captioner


CAPTION = (
    "A compact technical diagram has one wide rectangle centered in the middle, "
    "with a small circle above its left edge and even spacing between all outlined elements."
)


def _fake_cairosvg(monkeypatch):
    module = types.ModuleType("cairosvg")

    def svg2png(*, bytestring, write_to, output_width):
        assert bytestring.startswith(b"<svg")
        assert output_width == 800
        write_to.write(b"png")

    module.svg2png = svg2png
    monkeypatch.setitem(sys.modules, "cairosvg", module)


class _Client:
    def __init__(self, text=CAPTION):
        self.kwargs = None
        self.messages = SimpleNamespace(create=self._create)
        self.text = text

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)])


def test_caption_svg_builds_claude_request(monkeypatch):
    _fake_cairosvg(monkeypatch)
    client = _Client()

    out = captioner.caption_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>',
        {"page_title": "Page", "paper_title": "", "figure_index": 3, "ignored": "x"},
        client,
    )

    content = client.kwargs["messages"][0]["content"]
    assert out == CAPTION
    assert client.kwargs["model"] == "claude-sonnet-4-20250514"
    assert client.kwargs["max_tokens"] == 300
    assert content[0]["source"] == {"type": "base64", "media_type": "image/png", "data": "cG5n"}
    assert content[1]["text"] == 'Additional context: {"page_title": "Page", "figure_index": 3}'


@pytest.mark.parametrize("text,reason", [("", "empty_response"), ("too short", "response_too_short")])
def test_caption_svg_rejects_bad_responses(monkeypatch, text, reason):
    _fake_cairosvg(monkeypatch)
    with pytest.raises(captioner.CaptionError, match=reason):
        captioner.caption_svg("<svg/>", {}, _Client(text))


def test_batch_caption_returns_records_and_failures(monkeypatch):
    anthropic = types.ModuleType("anthropic")
    anthropic.Anthropic = lambda api_key: _Client()
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)
    monkeypatch.setattr(captioner, "caption_svg", lambda svg, metadata, client: CAPTION)
    monkeypatch.setattr(captioner.time, "sleep", lambda _: None)

    raw = SimpleNamespace(svg_string="<svg/>", source_id="a", domain="wiki", metadata={"page_title": "P"})
    bad = SimpleNamespace(svg_string="<bad", source_id="b", domain="wiki", metadata={})
    monkeypatch.setattr(
        "backend.dataset_pipeline.processing.normalizer.normalize_svg",
        lambda svg: (_ for _ in ()).throw(captioner.CaptionError("normalize_failed")) if svg == "<bad" else svg,
    )

    records, failures = captioner.batch_caption([raw, bad], SimpleNamespace(anthropic_api_key="key"))

    assert records == [
        captioner.TrainingRecord("a", CAPTION, "<svg/>", "wiki", "", {"page_title": "P"})
    ]
    assert failures == [(bad, "normalize_failed")]


def test_batch_caption_parallelism_preserves_input_order(monkeypatch):
    anthropic = types.ModuleType("anthropic")
    anthropic.Anthropic = lambda api_key: _Client()
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)

    def fake_caption(svg, metadata, client):
        if metadata.get("delay"):
            time.sleep(0.05)
        return f"{CAPTION} {metadata['label']}."

    monkeypatch.setattr(captioner, "caption_svg", fake_caption)

    slow = SimpleNamespace(
        svg_string="<svg/>",
        source_id="slow",
        domain="wiki",
        metadata={"label": "slow", "delay": True},
    )
    fast = SimpleNamespace(
        svg_string="<svg/>",
        source_id="fast",
        domain="wiki",
        metadata={"label": "fast"},
    )

    records, failures = captioner.batch_caption(
        [slow, fast],
        SimpleNamespace(anthropic_api_key="key", caption_parallelism=2),
        normalize=False,
    )

    assert failures == []
    assert [record.id for record in records] == ["slow", "fast"]
    assert [record.prompt for record in records] == [f"{CAPTION} slow.", f"{CAPTION} fast."]


def test_batch_caption_can_use_local_backend(monkeypatch):
    calls = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": CAPTION}}]}

    def fake_post(url, **kwargs):
        calls["url"] = url
        calls["payload"] = kwargs["json"]
        return _Response()

    monkeypatch.setattr(captioner.httpx, "post", fake_post)

    raw = SimpleNamespace(
        svg_string='<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>',
        source_id="local",
        domain="wiki",
        metadata={"page_title": "P"},
    )
    records, failures = captioner.batch_caption(
        [raw],
        SimpleNamespace(
            caption_backend="local",
            caption_local_base_url="http://localhost:11434/v1",
            caption_local_model="llava",
            caption_local_timeout_s=1.0,
            anthropic_api_key="",
        ),
        normalize=False,
    )

    assert failures == []
    assert records[0].id == "local"
    assert calls["url"].endswith("/chat/completions")
    assert calls["payload"]["model"] == "llava"
