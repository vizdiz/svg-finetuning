"""
Unit tests for dump-backed Wikimedia SVG discovery.

Run with:
    python -m pytest tests/test_wikimedia_dump_scraper.py -v
"""

from pathlib import Path

import httpx

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.scrapers import wikimedia_dump_scraper
from backend.dataset_pipeline.scrapers.wikimedia_dump_scraper import (
    WikimediaDumpScraper,
    iter_dump_candidates,
    rank_dump_candidates,
    score_dump_candidate,
    source_id,
    upload_url,
)


def _svg() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
        '<g><rect width="10" height="10"/></g>'
        "</svg>"
    )


def _insert_line(*rows: str) -> str:
    return "INSERT INTO `image` VALUES " + ",".join(rows) + ";\n"


def _row(
    filename: str,
    *,
    size: int = 1000,
    media_type: str = "DRAWING",
    major_mime: str = "image",
    minor_mime: str = "svg+xml",
) -> str:
    return (
        f"('{filename}',{size},100,100,'{{\"caption\":\"a,b\"}}',0,"
        f"'{media_type}','{major_mime}','{minor_mime}',1,2,'20250101000000','abc123')"
    )


class _FakeResponse:
    def __init__(self, text="", status_code=200, headers=None, url="https://example.com"):
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", self.url)
            response = httpx.Response(self.status_code, headers=self.headers, request=request)
            raise httpx.HTTPStatusError("failed", request=request, response=response)


class _FakeClient:
    responses = []
    calls = []

    def __init__(self, headers=None):
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        response.url = url
        return response


def _config(tmp_path: Path, dump_path: Path, **overrides) -> ScrapeConfig:
    values = {
        "max_svgs_per_source": 2,
        "min_svg_bytes": 0,
        "min_svg_elements": 1,
        "wikimedia_cache_dir": str(tmp_path / "cache"),
        "wikimedia_dump_path": str(dump_path),
        "wikimedia_dump_download_parallelism": 1,
        "wikimedia_dump_max_inflight": 1,
    }
    values.update(overrides)
    return ScrapeConfig(**values)


def test_iter_dump_candidates_filters_to_svg_rows(tmp_path):
    dump = tmp_path / "image.sql"
    dump.write_text(
        _insert_line(
            _row("Diagram, with comma.svg"),
            _row("Photo.jpg", minor_mime="jpeg"),
            _row("Tiny.svg", size=1),
        )
    )

    candidates = list(iter_dump_candidates(str(dump), ScrapeConfig(min_svg_bytes=10)))

    assert [candidate.filename for candidate in candidates] == ["Diagram, with comma.svg"]
    assert candidates[0].size_bytes == 1000
    assert candidates[0].timestamp == "20250101000000"


def test_gz_dump_is_decompressed_to_disk(tmp_path):
    import gzip

    dump_gz = tmp_path / "image.sql.gz"
    dump_sql = tmp_path / "image.sql"
    with gzip.open(dump_gz, "wt", encoding="utf-8") as fh:
        fh.write(_insert_line(_row("Diagram.svg")))

    candidates = list(iter_dump_candidates(str(dump_gz), ScrapeConfig(min_svg_bytes=10)))

    assert [candidate.filename for candidate in candidates] == ["Diagram.svg"]
    assert dump_sql.read_text().startswith("INSERT INTO `image` VALUES ")


def test_upload_url_uses_commons_hash_layout():
    filename = "Example diagram.svg"
    digest = "81df0a4e571c0b45f4900502e7a0c018"

    assert upload_url(filename, "https://upload.wikimedia.org/wikipedia/commons") == (
        f"https://upload.wikimedia.org/wikipedia/commons/{digest[0]}/{digest[:2]}/Example%20diagram.svg"
    )


def test_ranked_candidates_prioritize_diagram_like_names(tmp_path):
    dump = tmp_path / "image.sql"
    dump.write_text(
        _insert_line(
            _row("Company logo.svg", size=2000),
            _row("Network diagram.svg", size=2000),
        )
    )

    ranked = rank_dump_candidates(str(dump), ScrapeConfig(min_svg_bytes=10, wikimedia_dump_rank_multiplier=10))

    assert [candidate.filename for candidate in ranked[:2]] == ["Network diagram.svg", "Company logo.svg"]
    assert score_dump_candidate(ranked[0]) > score_dump_candidate(ranked[1])


def test_dump_scraper_downloads_and_caches_valid_svgs(monkeypatch, tmp_path):
    dump = tmp_path / "image.sql"
    dump.write_text(_insert_line(_row("Downloaded diagram.svg")))
    _FakeClient.responses = [_FakeResponse(text=_svg())]
    _FakeClient.calls = []
    monkeypatch.setattr(wikimedia_dump_scraper.httpx, "Client", _FakeClient)

    records = list(WikimediaDumpScraper().scrape(_config(tmp_path, dump)))

    sid = source_id("Downloaded diagram.svg")
    assert len(records) == 1
    assert records[0].source_id == sid
    assert records[0].domain == "wikipedia"
    assert records[0].metadata["discovery_source"] == "commons_image_dump"
    assert (tmp_path / "cache" / f"{sid}.svg").read_text() == _svg()
    assert len(_FakeClient.calls) == 1


def test_dump_scraper_downloads_in_rank_order(monkeypatch, tmp_path):
    dump = tmp_path / "image.sql"
    dump.write_text(
        _insert_line(
            _row("Company logo.svg", size=2000),
            _row("Network diagram.svg", size=2000),
        )
    )
    _FakeClient.responses = [_FakeResponse(text=_svg()), _FakeResponse(text=_svg())]
    _FakeClient.calls = []
    monkeypatch.setattr(wikimedia_dump_scraper.httpx, "Client", _FakeClient)

    list(
        WikimediaDumpScraper().scrape(
            _config(tmp_path, dump, max_svgs_per_source=2, wikimedia_dump_rank_multiplier=10)
        )
    )

    downloaded = [call[0] for call in _FakeClient.calls]
    assert downloaded[0].endswith("/Network%20diagram.svg")
    assert downloaded[1].endswith("/Company%20logo.svg")


def test_dump_scraper_uses_cache_without_downloading(monkeypatch, tmp_path):
    dump = tmp_path / "image.sql"
    dump.write_text(_insert_line(_row("Cached diagram.svg")))
    sid = source_id("Cached diagram.svg")
    cache_path = tmp_path / "cache" / f"{sid}.svg"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(_svg())
    _FakeClient.responses = []
    _FakeClient.calls = []
    monkeypatch.setattr(wikimedia_dump_scraper.httpx, "Client", _FakeClient)

    records = list(WikimediaDumpScraper().scrape(_config(tmp_path, dump)))

    assert len(records) == 1
    assert records[0].source_id == sid
    assert _FakeClient.calls == []
