"""
Integration tests for WikipediaScraper against a live subsample.

Run with:
    pytest tests/wikipedia_scraper.py -v

These hit real Wikimedia Commons endpoints; expect ~15-30s per test.
"""

import pytest
from lxml import etree

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.scrapers.wikipedia_scraper import WikipediaScraper, _CATEGORIES
from backend.dataset_pipeline.scrapers.base_scraper import RawSVG, count_svg_elements


@pytest.fixture(scope="module")
def subsample():
    config = ScrapeConfig(max_svgs_per_source=5)
    scraper = WikipediaScraper()
    return list(scraper.scrape(config))


@pytest.fixture(scope="module")
def config():
    return ScrapeConfig(max_svgs_per_source=5)


def test_yields_results(subsample):
    assert len(subsample) > 0, "Scraper yielded no SVGs"


def test_does_not_exceed_max_svgs(subsample):
    assert len(subsample) <= 5


def test_all_results_are_rawsvg(subsample):
    for item in subsample:
        assert isinstance(item, RawSVG)


def test_svg_strings_are_valid_xml(subsample):
    for item in subsample:
        etree.fromstring(item.svg_string.encode())


def test_no_embedded_raster_images(subsample):
    for item in subsample:
        root = etree.fromstring(item.svg_string.encode())
        assert not root.findall(".//{*}image"), f"{item.source_id} contains <image> tag"


def test_byte_size_within_bounds(subsample, config):
    for item in subsample:
        size = len(item.svg_string.encode())
        assert size >= config.min_svg_bytes, f"{item.source_id} too small: {size}B"
        assert size <= config.max_svg_bytes, f"{item.source_id} too large: {size}B"


def test_element_count_meets_minimum(subsample, config):
    for item in subsample:
        n = count_svg_elements(item.svg_string)
        assert n >= config.min_svg_elements, f"{item.source_id} has only {n} elements"


def test_domain_is_wikipedia(subsample):
    for item in subsample:
        assert item.domain == "wikipedia"


def test_source_id_format(subsample):
    import re
    for item in subsample:
        assert re.fullmatch(r"[a-z0-9_]{1,80}", item.source_id), (
            f"Unexpected source_id: {item.source_id!r}"
        )


def test_source_url_is_wikimedia(subsample):
    for item in subsample:
        assert "wikimedia.org" in item.source_url, (
            f"{item.source_id} unexpected source_url: {item.source_url}"
        )


def test_required_metadata_keys(subsample):
    required = {"page_title", "page_url", "category", "description", "file_size_bytes"}
    for item in subsample:
        assert required <= item.metadata.keys(), f"{item.source_id} missing metadata keys"


def test_category_is_known(subsample):
    for item in subsample:
        assert item.metadata["category"] in _CATEGORIES, (
            f"{item.source_id} has unknown category: {item.metadata['category']}"
        )


def test_page_url_is_commons(subsample):
    for item in subsample:
        assert item.metadata["page_url"].startswith("https://commons.wikimedia.org/"), (
            f"{item.source_id} unexpected page_url: {item.metadata['page_url']}"
        )


def test_no_duplicate_source_ids(subsample):
    ids = [item.source_id for item in subsample]
    assert len(ids) == len(set(ids)), "Duplicate source_ids found"
