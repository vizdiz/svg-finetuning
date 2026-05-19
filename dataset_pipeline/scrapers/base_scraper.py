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

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

import boto3
from lxml import etree

from dataset_pipeline.config import ScrapeConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RawSVG:
    svg_string: str
    source_url: str
    source_id: str
    domain: str
    metadata: dict = field(default_factory=dict)


def count_svg_elements(svg_string: str) -> int:
    root = etree.fromstring(svg_string.encode())
    return len(list(root))


class BaseScraper(ABC):
    @abstractmethod
    def scrape(self, config: ScrapeConfig) -> Iterator[RawSVG]:
        ...

    def upload_to_s3(self, raw_svg: RawSVG, bucket: str, prefix: str) -> str:
        key = f"{prefix}{raw_svg.domain}/{raw_svg.source_id}.svg"
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=key,
            Body=raw_svg.svg_string.encode(),
            ContentType="image/svg+xml",
        )
        uri = f"s3://{bucket}/{key}"
        logger.info("Uploaded %s", uri)
        return uri
