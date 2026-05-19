"""
dataset_pipeline.scrapers.base_scraper

Abstract base class for all SVG scrapers.

Defines the RawSVGRecord dataclass (raw SVG string, source URL, source name,
and optional metadata) and the BaseScraper interface that every scraper must
implement:

  - scrape() -> Iterator[RawSVGRecord]
      Yields raw records; implementations handle pagination, rate limiting,
      and retries internally.

  - source_name -> str
      Human-readable identifier for the source (e.g. "arxiv", "wikipedia").

Common retry logic, HTTP session management (httpx), and S3 upload helpers
are implemented here so scrapers don't duplicate them.
"""
