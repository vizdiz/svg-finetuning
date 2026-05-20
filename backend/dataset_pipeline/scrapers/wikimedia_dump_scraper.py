"""
dataset_pipeline.scrapers.wikimedia_dump_scraper

Streams Wikimedia Commons image table dumps to discover SVG files without
paginating the live Commons API. The dump supplies metadata only; raw SVG bytes
are downloaded from the canonical upload path, then cached locally.
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import logging
import math
import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

import httpx
from lxml import etree
from tqdm import tqdm

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.scrapers.base_scraper import BaseScraper, RawSVG, count_svg_elements
from backend.dataset_pipeline.scrapers.wikipedia_scraper import (
    _read_cached_svg,
    _retry_after_seconds,
    _slugify,
    _status_code,
    _write_cached_svg,
)

logger = logging.getLogger(__name__)

_RETRY_DELAYS = [1, 2, 4]
_SELECTED_IMAGE_COLUMNS = {0, 1, 2, 3, 6, 7, 8, 11, 12}


@dataclass(frozen=True)
class WikimediaDumpCandidate:
    filename: str
    size_bytes: int
    width: int
    height: int
    media_type: str
    major_mime: str
    minor_mime: str
    timestamp: str
    sha1: str


_POSITIVE_FILENAME_TOKENS = (
    "diagram",
    "flow",
    "chart",
    "graph",
    "network",
    "architecture",
    "pipeline",
    "schema",
    "schematic",
    "block",
    "system",
    "process",
    "sequence",
    "circuit",
    "map",
    "timeline",
    "tree",
    "uml",
    "topology",
    "logic",
)

_NEGATIVE_FILENAME_TOKENS = (
    "logo",
    "flag",
    "icon",
    "badge",
    "emblem",
    "seal",
    "favicon",
    "wordmark",
    "crest",
)


def _open_dump(path: str):
    if not path:
        raise RuntimeError(
            "wikimedia_dump_path is required for dump mode. Download "
            "https://dumps.wikimedia.org/commonswiki/latest/commonswiki-latest-image.sql.gz "
            "or set --wikimedia-dump-path to an existing image.sql(.gz) snapshot."
        )

    dump_path = Path(path)
    if not dump_path.exists():
        raise FileNotFoundError(f"Wikimedia dump not found: {dump_path}")

    if dump_path.suffix == ".gz":
        return _ensure_decompressed_dump(dump_path).open("rt", encoding="utf-8", errors="replace")
    return dump_path.open("rt", encoding="utf-8", errors="replace")


def _ensure_decompressed_dump(dump_path: Path) -> Path:
    if dump_path.suffix != ".gz":
        return dump_path

    decompressed_path = dump_path.with_suffix("")
    if decompressed_path.exists() and decompressed_path.stat().st_mtime >= dump_path.stat().st_mtime:
        return decompressed_path

    logger.info("Decompressing Wikimedia dump %s -> %s", dump_path, decompressed_path)
    decompressed_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = decompressed_path.with_suffix(decompressed_path.suffix + ".tmp")
    with gzip.open(dump_path, "rb") as src, tmp_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    tmp_path.replace(decompressed_path)
    return decompressed_path


def _parse_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _iter_insert_rows(line: str) -> Iterator[dict[int, str | None]]:
    values_at = line.find(" VALUES ")
    if values_at < 0:
        return

    i = values_at + len(" VALUES ")
    n = len(line)

    while i < n:
        while i < n and line[i] in " \t\r\n,;":
            i += 1
        if i >= n or line[i] != "(":
            break

        i += 1
        field_index = 0
        row: dict[int, str | None] = {}

        while i < n:
            collect = field_index in _SELECTED_IMAGE_COLUMNS
            token: list[str] = []

            if line[i] == "'":
                i += 1
                while i < n:
                    ch = line[i]
                    if ch == "\\" and i + 1 < n:
                        if collect:
                            token.append(line[i + 1])
                        i += 2
                        continue
                    if ch == "'":
                        i += 1
                        break
                    if collect:
                        token.append(ch)
                    i += 1
                value: str | None = "".join(token) if collect else None
            else:
                start = i
                while i < n and line[i] not in ",)":
                    i += 1
                raw = line[start:i].strip()
                value = None if raw.upper() == "NULL" else raw

            if collect:
                row[field_index] = value

            while i < n and line[i].isspace():
                i += 1
            if i >= n:
                break
            if line[i] == ",":
                field_index += 1
                i += 1
                continue
            if line[i] == ")":
                i += 1
                yield row
                break
            i += 1


def iter_dump_candidates(path: str, config: ScrapeConfig) -> Iterator[WikimediaDumpCandidate]:
    with _open_dump(path) as fh:
        for line in fh:
            if not line.startswith("INSERT INTO `image` VALUES "):
                continue
            for row in _iter_insert_rows(line):
                filename = row.get(0) or ""
                if not filename.lower().endswith(".svg"):
                    continue

                candidate = WikimediaDumpCandidate(
                    filename=filename,
                    size_bytes=_parse_int(row.get(1)),
                    width=_parse_int(row.get(2)),
                    height=_parse_int(row.get(3)),
                    media_type=row.get(6) or "",
                    major_mime=row.get(7) or "",
                    minor_mime=row.get(8) or "",
                    timestamp=row.get(11) or "",
                    sha1=row.get(12) or "",
                )

                if candidate.major_mime != "image" or candidate.minor_mime not in {"svg+xml", "svg"}:
                    continue
                if candidate.size_bytes < config.min_svg_bytes:
                    continue
                if candidate.size_bytes > config.max_svg_bytes:
                    continue
                if candidate.width <= 0 or candidate.height <= 0:
                    continue
                aspect_ratio = candidate.width / candidate.height
                if aspect_ratio < config.hard_min_aspect_ratio or aspect_ratio > config.hard_max_aspect_ratio:
                    continue
                yield candidate


def score_dump_candidate(candidate: WikimediaDumpCandidate) -> float:
    filename = candidate.filename.lower()
    stem = filename.rsplit(".", 1)[0]
    score = 0.0

    for token in _POSITIVE_FILENAME_TOKENS:
        if token in stem:
            score += 18.0

    for token in _NEGATIVE_FILENAME_TOKENS:
        if token in stem:
            score -= 35.0

    area = candidate.width * candidate.height
    score += min(35.0, max(0.0, math.log10(max(area, 1)) * 8.0))
    score += min(15.0, max(0.0, math.log10(max(candidate.size_bytes, 1)) * 4.0))

    aspect_ratio = candidate.width / candidate.height
    aspect_penalty = abs(math.log2(max(aspect_ratio, 1e-6)))
    score += max(0.0, 24.0 - aspect_penalty * 8.0)

    if 300 <= candidate.width <= 5000 and 300 <= candidate.height <= 5000:
        score += 10.0
    if candidate.width >= 800 or candidate.height >= 800:
        score += 4.0
    if candidate.width == candidate.height:
        score += 2.0

    return score


def rank_dump_candidates(path: str, config: ScrapeConfig) -> list[WikimediaDumpCandidate]:
    target = max(1, int(config.max_svgs_per_source * max(1, config.wikimedia_dump_rank_multiplier)))
    heap: list[tuple[float, str, WikimediaDumpCandidate]] = []
    seen = 0

    for candidate in iter_dump_candidates(path, config):
        seen += 1
        score = score_dump_candidate(candidate)
        item = (score, candidate.filename, candidate)
        if len(heap) < target:
            heapq.heappush(heap, item)
            continue
        if score > heap[0][0]:
            heapq.heapreplace(heap, item)

    ranked = [candidate for _, _, candidate in sorted(heap, key=lambda item: (-item[0], item[1]))]
    logger.info(
        "Ranked %d Wikimedia SVG candidates, keeping top %d for download",
        seen,
        len(ranked),
    )
    return ranked


def upload_url(filename: str, base_url: str) -> str:
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"{base_url.rstrip('/')}/{digest[0]}/{digest[:2]}/{quote(filename, safe='')}"


def source_id(filename: str) -> str:
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()[:12]
    stem = filename.rsplit(".", 1)[0]
    slug = _slugify(stem) or "svg"
    return f"commons_dump_{digest}_{slug}"[:120]


def _has_image_tag(svg_string: str) -> bool:
    try:
        root = etree.fromstring(svg_string.encode())
        return bool(root.findall(".//{*}image"))
    except etree.XMLSyntaxError:
        return True


def _is_valid_xml(svg_string: str) -> bool:
    try:
        etree.fromstring(svg_string.encode())
        return True
    except etree.XMLSyntaxError:
        return False


def _download_with_retry(client: httpx.Client, url: str, config: ScrapeConfig) -> str:
    for attempt, delay in enumerate(_RETRY_DELAYS, 1):
        try:
            resp = client.get(url, timeout=60)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            if attempt == len(_RETRY_DELAYS):
                raise
            retry_delay = delay
            if _status_code(exc) == 429:
                retry_delay = _retry_after_seconds(getattr(exc, "response", None), config.wikimedia_429_cooldown_s)
            logger.warning("Attempt %d failed for %s: %s; retrying in %ss", attempt, url, exc, retry_delay)
            time.sleep(retry_delay)


class WikimediaDumpScraper(BaseScraper):
    def scrape(self, config: ScrapeConfig) -> Iterator[RawSVG]:
        valid_count = 0
        filtered = 0
        failed_download = 0
        consecutive_429 = 0
        stop_after_429 = False
        rate_lock = threading.Lock()

        candidates = iter(rank_dump_candidates(config.wikimedia_dump_path, config))
        max_workers = max(1, int(config.wikimedia_dump_download_parallelism or 1))
        max_inflight = max(max_workers, int(config.wikimedia_dump_max_inflight or max_workers))
        bar = tqdm(desc="Wikimedia dump [0 valid / 0 filtered]", unit="svg")

        def _record_from_candidate(candidate: WikimediaDumpCandidate) -> RawSVG | None:
            nonlocal consecutive_429
            sid = source_id(candidate.filename)
            url = upload_url(candidate.filename, config.wikimedia_asset_base_url)
            svg_string = _read_cached_svg(config, sid)

            if svg_string is None:
                with httpx.Client(headers={"User-Agent": "svg-finetuning-dump-scraper/1.0"}) as client:
                    try:
                        svg_string = _download_with_retry(client, url, config)
                        with rate_lock:
                            consecutive_429 = 0
                    except Exception as exc:
                        if _status_code(exc) == 429:
                            with rate_lock:
                                consecutive_429 += 1
                        raise
                _write_cached_svg(config, sid, svg_string)

            if not _is_valid_xml(svg_string):
                return None
            if _has_image_tag(svg_string):
                return None
            if count_svg_elements(svg_string) < config.min_svg_elements:
                return None

            return RawSVG(
                svg_string=svg_string,
                source_url=url,
                source_id=sid,
                domain="wikipedia",
                metadata={
                    "page_title": f"File:{candidate.filename}",
                    "page_url": f"https://commons.wikimedia.org/wiki/File:{quote(candidate.filename, safe='')}",
                    "category": "commons_dump",
                    "description": "",
                    "file_size_bytes": candidate.size_bytes,
                    "width": candidate.width,
                    "height": candidate.height,
                    "timestamp": candidate.timestamp,
                    "sha1": candidate.sha1,
                    "discovery_source": "commons_image_dump",
                },
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            while valid_count < config.max_svgs_per_source and not stop_after_429:
                while len(futures) < max_inflight:
                    try:
                        candidate = next(candidates)
                    except StopIteration:
                        break
                    futures[pool.submit(_record_from_candidate, candidate)] = candidate

                if not futures:
                    break

                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    candidate = futures.pop(future)
                    if valid_count >= config.max_svgs_per_source:
                        break
                    try:
                        record = future.result()
                    except Exception as exc:
                        failed_download += 1
                        logger.warning("Failed Wikimedia dump SVG %s: %s", candidate.filename, exc)
                        with rate_lock:
                            should_stop = consecutive_429 >= config.wikimedia_max_consecutive_429
                        if should_stop:
                            logger.warning(
                                "Stopping Wikimedia dump scrape after %d consecutive 429 responses",
                                consecutive_429,
                            )
                            stop_after_429 = True
                            for pending in futures:
                                pending.cancel()
                            break
                        continue

                    if record is None:
                        filtered += 1
                    else:
                        valid_count += 1
                        bar.update(1)
                        yield record

                    bar.set_description(
                        f"Wikimedia dump [{valid_count} valid / {filtered} filtered / {failed_download} failed]"
                    )

        bar.close()
        logger.info(
            "Wikimedia dump scrape complete: %d valid SVGs, %d filtered, %d failed downloads",
            valid_count,
            filtered,
            failed_download,
        )
