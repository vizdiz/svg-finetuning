"""
Unit tests for Wikimedia scraper caching and throttling behavior.

Run with:
    python -m pytest tests/test_wikipedia_cache.py -v
"""

import httpx

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.scrapers import wikipedia_scraper
from backend.dataset_pipeline.scrapers.wikipedia_scraper import WikipediaScraper


def _svg() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
        '<rect width="10" height="10"/>'
        "</svg>"
    )


def _api_payload(*titles: str) -> dict:
    pages = {}
    for index, title in enumerate(titles):
        pages[str(index)] = {
            "title": f"File:{title}.svg",
            "imageinfo": [
                {
                    "mime": "image/svg+xml",
                    "url": f"https://upload.wikimedia.org/{title}.svg",
                    "size": len(_svg().encode()),
                }
            ],
        }
    return {"query": {"pages": pages}}


class _FakeResponse:
    def __init__(self, *, json_data=None, text="", status_code=200, headers=None, url="https://example.com"):
        self._json_data = json_data
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}
        self.url = url

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", self.url)
            response = httpx.Response(self.status_code, headers=self.headers, request=request)
            raise httpx.HTTPStatusError("failed", request=request, response=response)


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        response.url = url
        return response


def _config(tmp_path, **overrides) -> ScrapeConfig:
    values = {
        "max_svgs_per_source": 1,
        "min_svg_bytes": 0,
        "min_svg_elements": 1,
        "wikimedia_cache_dir": str(tmp_path),
        "wikimedia_download_delay_s": 0,
        "wikimedia_429_cooldown_s": 0,
    }
    values.update(overrides)
    return ScrapeConfig(**values)


def test_uses_cached_svg_without_downloading(monkeypatch, tmp_path):
    cache_path = tmp_path / "cached_titlesvg.svg"
    cache_path.write_text(_svg())
    fake_client = _FakeClient([_FakeResponse(json_data=_api_payload("Cached title"))])
    monkeypatch.setattr(wikipedia_scraper.httpx, "Client", lambda headers: fake_client)
    monkeypatch.setattr(wikipedia_scraper, "_CATEGORIES", ["Computer_network_diagrams"])
    monkeypatch.setattr(wikipedia_scraper.time, "sleep", lambda _: None)

    records = list(WikipediaScraper().scrape(_config(tmp_path)))

    assert len(records) == 1
    assert records[0].source_id == "cached_titlesvg"
    assert records[0].svg_string == _svg()
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0][0] == wikipedia_scraper._API


def test_writes_successful_download_to_cache(monkeypatch, tmp_path):
    fake_client = _FakeClient(
        [
            _FakeResponse(json_data=_api_payload("Downloaded title")),
            _FakeResponse(text=_svg()),
        ]
    )
    monkeypatch.setattr(wikipedia_scraper.httpx, "Client", lambda headers: fake_client)
    monkeypatch.setattr(wikipedia_scraper, "_CATEGORIES", ["Computer_network_diagrams"])
    monkeypatch.setattr(wikipedia_scraper.time, "sleep", lambda _: None)

    records = list(WikipediaScraper().scrape(_config(tmp_path)))

    assert len(records) == 1
    assert (tmp_path / "downloaded_titlesvg.svg").read_text() == _svg()
    assert [call[0] for call in fake_client.calls] == [
        wikipedia_scraper._API,
        "https://upload.wikimedia.org/Downloaded title.svg",
    ]


def test_stops_after_consecutive_429s(monkeypatch, tmp_path):
    fake_client = _FakeClient(
        [
            _FakeResponse(json_data=_api_payload("One", "Two", "Three")),
            _FakeResponse(status_code=429, headers={"Retry-After": "0"}),
            _FakeResponse(status_code=429, headers={"Retry-After": "0"}),
            _FakeResponse(status_code=429, headers={"Retry-After": "0"}),
            _FakeResponse(status_code=429, headers={"Retry-After": "0"}),
            _FakeResponse(status_code=429, headers={"Retry-After": "0"}),
            _FakeResponse(status_code=429, headers={"Retry-After": "0"}),
        ]
    )
    sleep_calls = []
    monkeypatch.setattr(wikipedia_scraper.httpx, "Client", lambda headers: fake_client)
    monkeypatch.setattr(wikipedia_scraper, "_CATEGORIES", ["Computer_network_diagrams"])
    monkeypatch.setattr(wikipedia_scraper, "_RETRY_DELAYS", [0, 0, 0])
    monkeypatch.setattr(wikipedia_scraper.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    records = list(
        WikipediaScraper().scrape(
            _config(
                tmp_path,
                max_svgs_per_source=10,
                wikimedia_max_consecutive_429=2,
            )
        )
    )

    assert records == []
    downloaded_urls = [url for url, _ in fake_client.calls if url != wikipedia_scraper._API]
    assert downloaded_urls == [
        "https://upload.wikimedia.org/One.svg",
        "https://upload.wikimedia.org/One.svg",
        "https://upload.wikimedia.org/One.svg",
        "https://upload.wikimedia.org/Two.svg",
        "https://upload.wikimedia.org/Two.svg",
        "https://upload.wikimedia.org/Two.svg",
    ]
    assert not (tmp_path / "three.svg").exists()
