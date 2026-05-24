from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.corpus.fetch_arxiv_id_index import fetch_arxiv_ids
from tools.corpus.fetch_github_svg_index import fetch_github_svg_index


_ATOM_RESPONSE = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>3</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v2</id>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00003v1</id>
  </entry>
</feed>"""


def test_fetch_arxiv_ids_basic():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = _ATOM_RESPONSE
    mock_resp.raise_for_status = lambda: None

    with patch("httpx.get", return_value=mock_resp):
        with patch("tools.corpus.fetch_arxiv_id_index.time") as mock_time:
            mock_time.sleep = lambda _: None
            ids = fetch_arxiv_ids(
                categories=["cs.CV"],
                limit=10,
                start_date="2024-01-01",
                end_date="2024-12-31",
            )

    assert len(ids) == 3
    assert "2401.00001" in ids
    assert "2401.00002" in ids
    assert "2401.00003" in ids
    # Version suffix stripped
    for i in ids:
        assert "v" not in i


def test_fetch_arxiv_ids_respects_limit():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = _ATOM_RESPONSE
    mock_resp.raise_for_status = lambda: None

    with patch("httpx.get", return_value=mock_resp):
        with patch("tools.corpus.fetch_arxiv_id_index.time") as mock_time:
            mock_time.sleep = lambda _: None
            ids = fetch_arxiv_ids(
                categories=["cs.CV"],
                limit=2,
                start_date=None,
                end_date=None,
            )

    assert len(ids) == 2


_GH_TREE_RESPONSE = {
    "tree": [
        {"type": "blob", "path": "docs/diagram.svg", "sha": "abc123"},
        {"type": "blob", "path": "arch.svg", "sha": "def456"},
        {"type": "blob", "path": "README.md", "sha": "ghi789"},
        {"type": "tree", "path": "src", "sha": "zzz"},
    ]
}


def test_fetch_github_svg_index_tree_mode():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _GH_TREE_RESPONSE
    mock_resp.raise_for_status = lambda: None

    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = mock_resp

    with patch("httpx.Client", return_value=mock_client):
        with patch("tools.corpus.fetch_github_svg_index.time") as mock_time:
            mock_time.sleep = lambda _: None
            # Only first repo is needed, limit=2
            rows = fetch_github_svg_index(limit=2, token=None)

    assert len(rows) == 2
    assert all(r["raw_url"].startswith("https://raw.githubusercontent.com") for r in rows)
    assert all(r["path"].endswith(".svg") for r in rows)
    assert rows[0]["sha"] == "abc123"
    assert rows[1]["sha"] == "def456"


def test_fetch_github_svg_index_dedupes_by_sha():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _GH_TREE_RESPONSE  # same tree for all repos
    mock_resp.raise_for_status = lambda: None

    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = mock_resp

    with patch("httpx.Client", return_value=mock_client):
        with patch("tools.corpus.fetch_github_svg_index.time") as mock_time:
            mock_time.sleep = lambda _: None
            # Both repos return same SHA — should dedupe
            rows = fetch_github_svg_index(limit=100, token=None)

    shas = [r["sha"] for r in rows]
    assert len(shas) == len(set(shas))  # no duplicates
