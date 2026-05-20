"""
dataset_pipeline.scrapers.wikipedia_scraper

Scrapes SVG files from Wikimedia Commons.

Queries the Wikimedia Commons API for files in the category tree rooted at
"SVG files" (e.g. SVG diagrams, SVG maps, SVG flags, SVG charts), downloads
each file's raw SVG content, and yields RawSVGRecord objects with the file's
page title, description, and Commons URL as metadata.

Uses the file description page to extract human-written captions and
categories, which are passed through to the captioner as prior context
to improve caption quality.

Handles:
  - Paginated API responses (cmcontinue)
  - Skipping non-SVG MIME types returned by the category API
  - Respecting Wikimedia's rate limits via httpx with backoff
"""

import logging
import re
import time
from pathlib import Path
from typing import Iterator

import httpx
from tqdm import tqdm

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.scrapers.base_scraper import BaseScraper, RawSVG, count_svg_elements

logger = logging.getLogger(__name__)

_API = "https://commons.wikimedia.org/w/api.php"
_CATEGORIES = [
    "Computer_network_diagrams",
    "Computer_architecture_diagrams",
    "Computing_diagrams",
    "Electronics_diagrams",
    "Engineering_diagrams",
    "Scientific_diagrams",
]
_RATE_LIMIT_S = 1
_RETRY_DELAYS = [1, 2, 4]


def _slugify(title: str) -> str:
    slug = title.lower().replace(" ", "_")
    slug = re.sub(r"[^\w]", "", slug)
    return slug[:80]


def _retry_after_seconds(resp: httpx.Response | None, fallback: float) -> float:
    if resp is None:
        return fallback
    value = resp.headers.get("Retry-After")
    if not value:
        return fallback
    try:
        return max(float(value), fallback)
    except ValueError:
        return fallback


def _status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _cache_path(config: ScrapeConfig, source_id: str) -> Path:
    return Path(config.wikimedia_cache_dir) / f"{source_id}.svg"


def _read_cached_svg(config: ScrapeConfig, source_id: str) -> str | None:
    path = _cache_path(config, source_id)
    if not path.exists():
        return None
    try:
        return path.read_text()
    except OSError as exc:
        logger.warning("Failed to read Wikimedia cache %s: %s", path, exc)
        return None


def _write_cached_svg(config: ScrapeConfig, source_id: str, svg_string: str) -> None:
    path = _cache_path(config, source_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(svg_string)
        tmp.replace(path)
    except OSError as exc:
        logger.warning("Failed to write Wikimedia cache %s: %s", path, exc)


def _fetch_with_retry(client: httpx.Client, url: str, **kwargs) -> httpx.Response:
    for attempt, delay in enumerate(_RETRY_DELAYS, 1):
        try:
            resp = client.get(url, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            if attempt == len(_RETRY_DELAYS):
                raise
            retry_delay = delay
            if _status_code(exc) == 429:
                retry_delay = _retry_after_seconds(getattr(exc, "response", None), delay)
            logger.warning("Attempt %d failed for %s: %s — retrying in %ss", attempt, url, exc, retry_delay)
            time.sleep(retry_delay)


def _has_image_tag(svg_string: str) -> bool:
    from lxml import etree
    try:
        root = etree.fromstring(svg_string.encode())
        return bool(root.findall(".//{*}image"))
    except etree.XMLSyntaxError:
        return True


def _is_valid_xml(svg_string: str) -> bool:
    from lxml import etree
    try:
        etree.fromstring(svg_string.encode())
        return True
    except etree.XMLSyntaxError:
        return False


class WikipediaScraper(BaseScraper):
    def scrape(self, config: ScrapeConfig) -> Iterator[RawSVG]:
        valid_count = 0
        filtered = 0
        consecutive_429 = 0
        seen: set[str] = set()

        bar = tqdm(desc=f"Wikipedia [{valid_count} valid / {filtered} filtered]", unit="svg")

        with httpx.Client(headers={"User-Agent": "svg-finetuning-scraper/1.0"}) as client:
            for category in _CATEGORIES:
                if valid_count >= config.max_svgs_per_source:
                    break

                logger.info("Scraping category: %s", category)
                continue_token: dict = {}

                while valid_count < config.max_svgs_per_source:
                    params = {
                        "action": "query",
                        "generator": "categorymembers",
                        "gcmtype": "file",
                        "gcmlimit": "50",
                        "gcmtitle": f"Category:{category}",
                        "prop": "imageinfo",
                        "iiprop": "url|size|mime",
                        "format": "json",
                        **continue_token,
                    }

                    try:
                        resp = _fetch_with_retry(client, _API, params=params)
                    except Exception as exc:
                        logger.warning("API error for category %s: %s", category, exc)
                        break

                    data = resp.json()
                    pages = data.get("query", {}).get("pages", {})

                    for page in pages.values():
                        if valid_count >= config.max_svgs_per_source:
                            break

                        imageinfo = page.get("imageinfo", [])
                        if not imageinfo:
                            continue

                        info = imageinfo[0]
                        if info.get("mime") != "image/svg+xml":
                            continue

                        page_title = page.get("title", "")
                        source_id = _slugify(page_title.removeprefix("File:"))
                        if source_id in seen:
                            continue

                        svg_url = info.get("url", "")
                        if not svg_url:
                            continue

                        svg_string = _read_cached_svg(config, source_id)
                        if svg_string is None:
                            try:
                                svg_resp = _fetch_with_retry(client, svg_url)
                                consecutive_429 = 0
                            except Exception as exc:
                                if _status_code(exc) == 429:
                                    consecutive_429 += 1
                                    cooldown = _retry_after_seconds(
                                        getattr(exc, "response", None),
                                        config.wikimedia_429_cooldown_s,
                                    )
                                    logger.warning(
                                        "Wikimedia 429 for %s (%d consecutive); cooling down for %ss",
                                        svg_url,
                                        consecutive_429,
                                        cooldown,
                                    )
                                    time.sleep(cooldown)
                                    if consecutive_429 >= config.wikimedia_max_consecutive_429:
                                        logger.warning(
                                            "Stopping Wikimedia scrape after %d consecutive 429 responses",
                                            consecutive_429,
                                        )
                                        valid_count = config.max_svgs_per_source
                                        break
                                logger.warning("Failed to download %s: %s", svg_url, exc)
                                filtered += 1
                                bar.set_description(f"Wikipedia [{valid_count} valid / {filtered} filtered]")
                                continue
                            finally:
                                time.sleep(config.wikimedia_download_delay_s)

                            svg_string = svg_resp.text
                            _write_cached_svg(config, source_id, svg_string)
                        else:
                            logger.info("Using cached Wikimedia SVG for %s", source_id)

                        svg_bytes = svg_string.encode()

                        if len(svg_bytes) < config.min_svg_bytes:
                            filtered += 1
                            bar.set_description(f"Wikipedia [{valid_count} valid / {filtered} filtered]")
                            continue
                        if len(svg_bytes) > config.max_svg_bytes:
                            filtered += 1
                            bar.set_description(f"Wikipedia [{valid_count} valid / {filtered} filtered]")
                            continue
                        if not _is_valid_xml(svg_string):
                            filtered += 1
                            bar.set_description(f"Wikipedia [{valid_count} valid / {filtered} filtered]")
                            continue
                        if _has_image_tag(svg_string):
                            filtered += 1
                            bar.set_description(f"Wikipedia [{valid_count} valid / {filtered} filtered]")
                            continue
                        if count_svg_elements(svg_string) < config.min_svg_elements:
                            filtered += 1
                            bar.set_description(f"Wikipedia [{valid_count} valid / {filtered} filtered]")
                            continue

                        seen.add(source_id)
                        yield RawSVG(
                            svg_string=svg_string,
                            source_url=svg_url,
                            source_id=source_id,
                            domain="wikipedia",
                            metadata={
                                "page_title": page_title,
                                "page_url": f"https://commons.wikimedia.org/wiki/{page_title.replace(' ', '_')}",
                                "category": category,
                                "description": info.get("extmetadata", {}).get("ImageDescription", {}).get("value", ""),
                                "file_size_bytes": info.get("size", 0),
                            },
                        )

                        valid_count += 1
                        bar.set_description(f"Wikipedia [{valid_count} valid / {filtered} filtered]")
                        bar.update(1)

                    continue_token = data.get("continue", {})
                    if not continue_token:
                        break

                    time.sleep(_RATE_LIMIT_S)

        bar.close()
        logger.info("Wikipedia scrape complete: %d valid SVGs, %d filtered", valid_count, filtered)
