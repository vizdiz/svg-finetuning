"""
Integration tests for ArxivScraper against a live subsample.

Run with:
    pytest tests/arxiv_scraper.py -v

These hit real arXiv endpoints; expect ~30-60s per test.
"""

import pytest
from lxml import etree

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.scrapers.arxiv_scraper import ArxivScraper
from backend.dataset_pipeline.scrapers.base_scraper import RawSVG, count_svg_elements


@pytest.fixture(scope="module")
def subsample():
    config = ScrapeConfig(max_svgs_per_source=5)
    scraper = ArxivScraper()
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


def test_domain_is_arxiv(subsample):
    for item in subsample:
        assert item.domain == "arxiv"


def test_source_url_is_arxiv_pdf(subsample):
    for item in subsample:
        assert item.source_url.startswith("https://arxiv.org/pdf/"), (
            f"{item.source_id} has unexpected source_url: {item.source_url}"
        )


def test_source_id_format(subsample):
    for item in subsample:
        parts = item.source_id.rsplit("_", 1)
        assert len(parts) == 2 and parts[1].isdigit(), f"Unexpected source_id: {item.source_id}"


def test_required_metadata_keys(subsample):
    required = {"paper_title", "paper_abstract", "figure_index", "paper_url", "categories", "raster_skip_count"}
    for item in subsample:
        assert required <= item.metadata.keys(), f"{item.source_id} missing metadata keys"


def test_figure_index_matches_source_id(subsample):
    for item in subsample:
        page_index = int(item.source_id.rsplit("_", 1)[1])
        assert item.metadata["figure_index"] == page_index, (
            f"{item.source_id}: figure_index {item.metadata['figure_index']} != source_id suffix {page_index}"
        )


def test_metadata_categories_is_list(subsample):
    for item in subsample:
        assert isinstance(item.metadata["categories"], list)


def test_raster_skip_count_is_non_negative(subsample):
    for item in subsample:
        assert isinstance(item.metadata["raster_skip_count"], int)
        assert item.metadata["raster_skip_count"] >= 0
