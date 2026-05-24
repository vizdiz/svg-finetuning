from __future__ import annotations

import gzip
import io
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.dataset_pipeline.corpus.commoncrawl_index import (
    _parse_cdxj_line,
    _svg_score,
    fetch_commoncrawl_index_manifest,
    iter_commoncrawl_index_rows,
)


def _make_cdx_gz(rows: list[dict]) -> bytes:
    lines = "\n".join(json.dumps(r) for r in rows)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(lines.encode())
    return buf.getvalue()


def _stream_mock(status_code: int, payload: bytes):
    """Build a context-manager mock for httpx.stream() that yields payload in one chunk."""

    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_bytes = MagicMock(return_value=iter([payload]))

    @contextmanager
    def _cm(*args, **kwargs):
        yield mock_resp

    return _cm


def test_parse_cdxj_line_json_object():
    row = {"url": "https://example.com/a.svg", "mime": "image/svg+xml"}
    result = _parse_cdxj_line(json.dumps(row))
    assert result == row


def test_parse_cdxj_line_cdxj_format():
    payload = {"mime": "image/svg+xml", "status": "200"}
    line = f"com,example)/a.svg 20260101120000 {json.dumps(payload)}"
    result = _parse_cdxj_line(line)
    assert result is not None
    assert result["mime"] == "image/svg+xml"
    assert result["urlkey"] == "com,example)/a.svg"
    assert result["timestamp"] == "20260101120000"


def test_parse_cdxj_line_empty():
    assert _parse_cdxj_line("") is None
    assert _parse_cdxj_line("   ") is None


def test_svg_score_high_for_svg_url_and_mime():
    row = {"url": "https://example.com/diagram.svg", "mime": "image/svg+xml", "status": "200"}
    score = _svg_score(row)
    assert score >= 0.9


def test_svg_score_low_for_non_svg():
    row = {"url": "https://example.com/image.png", "mime": "image/png"}
    score = _svg_score(row)
    assert score < 0.3


def test_iter_commoncrawl_index_rows_https_path():
    svg_row = {"url": "https://example.com/fig.svg", "mime": "image/svg+xml", "status": "200",
               "filename": "crawl-data/CC-MAIN-2026-17/warc/foo.warc.gz", "offset": "100", "length": "200"}
    non_svg = {"url": "https://example.com/page.html", "mime": "text/html"}
    payload = _make_cdx_gz([svg_row, non_svg])

    with patch(
        "backend.dataset_pipeline.corpus.commoncrawl_index.httpx.stream",
        _stream_mock(200, payload),
    ):
        rows = list(
            iter_commoncrawl_index_rows(
                crawl_id="CC-MAIN-2026-17",
                index_keys=["cc-index/collections/CC-MAIN-2026-17/indexes/cdx-00000.gz"],
            )
        )

    assert len(rows) == 1
    assert rows[0]["url"] == "https://example.com/fig.svg"
    assert "warc_uri" in rows[0]
    assert rows[0]["warc_uri"].startswith("s3://commoncrawl/")
    assert rows[0]["score"] >= 0.9
    assert rows[0]["crawl_id"] == "CC-MAIN-2026-17"


def test_iter_commoncrawl_index_rows_skips_404():
    with patch(
        "backend.dataset_pipeline.corpus.commoncrawl_index.httpx.stream",
        _stream_mock(404, b""),
    ):
        rows = list(
            iter_commoncrawl_index_rows(
                crawl_id="CC-MAIN-2026-17",
                index_keys=["cc-index/collections/CC-MAIN-2026-17/indexes/cdx-00000.gz"],
            )
        )

    assert rows == []


def test_iter_commoncrawl_index_rows_limit():
    svg_row = {"url": "https://example.com/a.svg", "mime": "image/svg+xml"}
    payload = _make_cdx_gz([svg_row] * 10)

    with patch(
        "backend.dataset_pipeline.corpus.commoncrawl_index.httpx.stream",
        _stream_mock(200, payload),
    ):
        rows = list(
            iter_commoncrawl_index_rows(
                crawl_id="CC-MAIN-2026-17",
                index_keys=["key1.gz", "key2.gz"],
                limit=3,
            )
        )

    assert len(rows) == 3


def test_fetch_commoncrawl_index_manifest_writes_files(tmp_path):
    svg_row = {"url": "https://example.com/diagram.svg", "mime": "image/svg+xml", "status": "200",
               "filename": "crawl/warc/foo.warc.gz"}
    payload = _make_cdx_gz([svg_row])

    out = tmp_path / "index.jsonl"
    with patch(
        "backend.dataset_pipeline.corpus.commoncrawl_index.httpx.stream",
        _stream_mock(200, payload),
    ):
        summary = fetch_commoncrawl_index_manifest(
            crawl_id="CC-MAIN-2026-17",
            output_path=out,
            index_keys=["cc-index/collections/CC-MAIN-2026-17/indexes/cdx-00000.gz"],
        )

    assert out.exists()
    assert summary["record_count"] == 1
    assert (tmp_path / "index.summary.json").exists()
    row = json.loads(out.read_text().strip())
    assert row["url"] == "https://example.com/diagram.svg"
