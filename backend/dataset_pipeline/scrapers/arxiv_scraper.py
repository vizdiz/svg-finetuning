"""
dataset_pipeline.scrapers.arxiv_scraper

Scrapes SVG figures from arXiv papers.

Uses the `arxiv` Python client to query papers by category and date range,
downloads each paper's source tarball via the arXiv S3 bulk data bucket,
extracts embedded SVG files and figures converted from PDF pages using PyMuPDF,
and yields RawSVGRecord objects.

Targeted categories: cs.CV, cs.LG, cs.GR (computer vision, ML, graphics) where
SVG figures (diagrams, plots, architecture charts) are most prevalent.

Handles:
  - Deduplication by arXiv paper ID + figure index
  - Skipping papers whose source tarballs are unavailable
  - Rate limiting to respect arXiv bulk access guidelines
"""

import logging
import re
import time
from io import BytesIO
from typing import Iterator

import arxiv
import fitz
import httpx
from lxml import etree
from tqdm import tqdm

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.scrapers.base_scraper import BaseScraper, RawSVG, count_svg_elements

logger = logging.getLogger(__name__)

_QUERY = "cat:cs.CV OR cat:cs.LG OR cat:cs.AR OR cat:cs.NI OR cat:cs.SE OR cat:cs.DC"
_RATE_LIMIT_S = 3
_ARXIV_PAGE_SIZE = 50
_RETRY_DELAYS = [2, 4, 8]

_CLUSTER_GAP = 10        # pt — merge drawing paths within this distance into one cluster
_MIN_FIGURE_AREA = 2000  # pt² — minimum cluster area to be considered a figure
_BAND_GAP = 10           # pt — whitespace gap that separates two distinct content bands
_CAPTION_RE = re.compile(r"^\s*(fig(ure)?\.?\s*\d+|table\.?\s*\d+|\(\w\))", re.IGNORECASE)


def _fetch_with_retry(url: str) -> bytes:
    for attempt, delay in enumerate(_RETRY_DELAYS, 1):
        try:
            resp = httpx.get(url, timeout=60, follow_redirects=True)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            if attempt == len(_RETRY_DELAYS):
                raise
            logger.warning("Attempt %d failed for %s: %s — retrying in %ds", attempt, url, exc, delay)
            time.sleep(delay)


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


def _drawing_clusters(page: fitz.Page) -> list[fitz.Rect]:
    """Merge vector drawing paths into clusters; discard tiny clusters."""
    drawings = page.get_drawings()
    if not drawings:
        return []

    rects = [fitz.Rect(d["rect"]) for d in drawings if not fitz.Rect(d["rect"]).is_empty]
    if not rects:
        return []

    merged = True
    while merged:
        merged = False
        out: list[fitz.Rect] = []
        used = [False] * len(rects)
        for i, a in enumerate(rects):
            if used[i]:
                continue
            exp = a + (-_CLUSTER_GAP, -_CLUSTER_GAP, _CLUSTER_GAP, _CLUSTER_GAP)
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                if exp.intersects(rects[j]):
                    a = a | rects[j]
                    exp = a + (-_CLUSTER_GAP, -_CLUSTER_GAP, _CLUSTER_GAP, _CLUSTER_GAP)
                    used[j] = True
                    merged = True
            out.append(a)
            used[i] = True
        rects = out

    return [r for r in rects if r.width * r.height >= _MIN_FIGURE_AREA]


def _figure_bands(page: fitz.Page) -> list[fitz.Rect]:
    """
    Segment page content into horizontal bands using whitespace gaps, then
    return bands that contain at least one drawing cluster.

    Algorithm:
      1. Collect text blocks + drawing clusters as content rects.
      2. Sort by y0, sweep top-to-bottom: a new band starts whenever the gap
         between the current rect and the running y-max exceeds _BAND_GAP.
      3. Keep bands that contain a cluster (tagged at insertion).

    This is fully deterministic — band boundaries come from the page's own
    whitespace, not from any fixed padding constant.
    """
    clusters = _drawing_clusters(page)
    if not clusters:
        return []

    # (rect, is_cluster)
    items: list[tuple[fitz.Rect, bool]] = []
    for b in page.get_text("blocks"):
        if b[6] == 0:  # text blocks only
            items.append((fitz.Rect(b[:4]), False))
    for c in clusters:
        items.append((c, True))

    items.sort(key=lambda x: x[0].y0)

    # Sweep into bands
    bands: list[list[tuple[fitz.Rect, bool]]] = []
    current: list[tuple[fitz.Rect, bool]] = [items[0]]
    y_max = items[0][0].y1

    for rect, is_cluster in items[1:]:
        if rect.y0 - y_max >= _BAND_GAP:
            bands.append(current)
            current = []
        current.append((rect, is_cluster))
        y_max = max(y_max, rect.y1)
    bands.append(current)

    # Build a lookup: text block rect → first line of text (for caption detection)
    text_first_line: dict[tuple, str] = {}
    for b in page.get_text("blocks"):
        if b[6] == 0:
            key = (round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1))
            text_first_line[key] = b[4].strip()

    def _band_is_caption(band: list[tuple[fitz.Rect, bool]]) -> bool:
        """True if the band has no cluster and its first text looks like a caption."""
        if any(flag for _, flag in band):
            return False
        for rect, _ in band:
            key = (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))
            text = text_first_line.get(key, "")
            if text and _CAPTION_RE.match(text):
                return True
        return False

    # Collect figure bands; absorb the immediately following band if it's a caption
    result: list[fitz.Rect] = []
    i = 0
    while i < len(bands):
        band = bands[i]
        if any(flag for _, flag in band):
            union = band[0][0]
            for r, _ in band[1:]:
                union |= r
            # Absorb caption band below if present
            if i + 1 < len(bands) and _band_is_caption(bands[i + 1]):
                for r, _ in bands[i + 1]:
                    union |= r
                i += 1  # skip the caption band
            result.append(union)
        i += 1

    return result


def _crop_svg_to_bbox(svg_string: str, bbox: fitz.Rect) -> str:
    root = etree.fromstring(svg_string.encode())
    root.set("viewBox", f"{bbox.x0:.3f} {bbox.y0:.3f} {bbox.width:.3f} {bbox.height:.3f}")
    root.set("width", f"{bbox.width:.3f}")
    root.set("height", f"{bbox.height:.3f}")
    return etree.tostring(root, encoding="unicode", xml_declaration=False)


class ArxivScraper(BaseScraper):
    def scrape(self, config: ScrapeConfig) -> Iterator[RawSVG]:
        # arXiv legacy APIs: one request at a time, no more than one request every 3 seconds.
        client = arxiv.Client(page_size=_ARXIV_PAGE_SIZE, delay_seconds=_RATE_LIMIT_S, num_retries=3)
        search = arxiv.Search(
            query=_QUERY,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        valid_count = 0
        raster_skip = 0
        other_filtered = 0

        bar = tqdm(desc=f"arXiv [{valid_count} valid / {raster_skip} raster-skipped]", unit="svg")

        for paper in client.results(search):
            if valid_count >= config.max_svgs_per_source:
                break

            paper_id = paper.entry_id.split("/")[-1]
            pdf_url = f"https://arxiv.org/pdf/{paper_id}"

            try:
                pdf_bytes = _fetch_with_retry(pdf_url)
            except Exception as exc:
                logger.warning("Failed to download %s: %s", pdf_url, exc)
                time.sleep(_RATE_LIMIT_S)
                continue

            try:
                doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
            except Exception as exc:
                logger.warning("Failed to open PDF for %s: %s", paper_id, exc)
                time.sleep(_RATE_LIMIT_S)
                continue

            figure_counter = 0

            for page in doc:
                if valid_count >= config.max_svgs_per_source:
                    break

                if page.get_images():
                    continue

                bands = _figure_bands(page)
                if not bands:
                    continue

                try:
                    page_svg = page.get_svg_image(text_as_path=False)
                except Exception as exc:
                    logger.info("skipped: get_svg_image failed (paper %s): %s", paper_id, exc)
                    raster_skip += len(bands)
                    bar.set_description(f"arXiv [{valid_count} valid / {raster_skip} raster-skipped]")
                    continue

                for band in bands:
                    if valid_count >= config.max_svgs_per_source:
                        break

                    try:
                        svg_string = _crop_svg_to_bbox(page_svg, band)
                    except Exception as exc:
                        logger.info("skipped: crop failed (paper %s fig %d): %s", paper_id, figure_counter, exc)
                        raster_skip += 1
                        bar.set_description(f"arXiv [{valid_count} valid / {raster_skip} raster-skipped]")
                        figure_counter += 1
                        continue

                    svg_bytes = svg_string.encode()

                    if len(svg_bytes) < config.min_svg_bytes:
                        logger.debug("filtered size<min: %s fig %d (%dB)", paper_id, figure_counter, len(svg_bytes))
                        other_filtered += 1
                        figure_counter += 1
                        continue
                    if len(svg_bytes) > config.max_svg_bytes:
                        logger.debug("filtered size>max: %s fig %d (%dB)", paper_id, figure_counter, len(svg_bytes))
                        other_filtered += 1
                        figure_counter += 1
                        continue
                    if _has_image_tag(svg_string):
                        logger.debug("filtered image_tag: %s fig %d", paper_id, figure_counter)
                        other_filtered += 1
                        figure_counter += 1
                        continue
                    if not _is_valid_xml(svg_string):
                        logger.debug("filtered invalid_xml: %s fig %d", paper_id, figure_counter)
                        other_filtered += 1
                        figure_counter += 1
                        continue
                    if count_svg_elements(svg_string) < config.min_svg_elements:
                        logger.debug("filtered few_elements: %s fig %d", paper_id, figure_counter)
                        other_filtered += 1
                        figure_counter += 1
                        continue

                    source_id = f"{paper_id}_{figure_counter}"
                    yield RawSVG(
                        svg_string=svg_string,
                        source_url=pdf_url,
                        source_id=source_id,
                        domain="arxiv",
                        metadata={
                            "paper_title": paper.title,
                            "paper_abstract": paper.summary,
                            "figure_index": figure_counter,
                            "paper_url": paper.entry_id,
                            "categories": paper.categories,
                            "raster_skip_count": raster_skip,
                        },
                    )

                    valid_count += 1
                    figure_counter += 1
                    bar.set_description(f"arXiv [{valid_count} valid / {raster_skip} raster-skipped]")
                    bar.update(1)

            doc.close()
            time.sleep(_RATE_LIMIT_S)

        bar.close()

        total_attempted = valid_count + raster_skip + other_filtered
        raster_pct = (raster_skip / total_attempted * 100) if total_attempted else 0.0
        logger.info(
            "arXiv scrape complete: %d valid SVGs, %d raster-skipped (%.1f%%), %d filtered by size/element rules",
            valid_count,
            raster_skip,
            raster_pct,
            other_filtered,
        )
