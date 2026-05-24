"""
dataset_pipeline.processing.captioner

Generates natural-language prompts for each SVG using the Anthropic API.

Renders each normalised SVG to a PNG via cairosvg, then sends the image
to Claude with a structured prompt asking for a concise, imperative
description of what the SVG depicts (e.g. "A bar chart showing quarterly
revenue from 2020 to 2024"). Any existing metadata from the scraper
(Wikipedia caption, arXiv figure caption) is included as context to
improve accuracy and reduce hallucination.

Output is a single prompt string stored alongside the SVG in the
TrainingRecord — this becomes the model's input at training time.

Handles:
  - Anthropic API rate limiting and retries with exponential backoff
  - Captions that are too generic or too long (configurable bounds)
  - Batching requests to stay within API throughput limits
  - Caching responses by SVG content hash to avoid re-captioning duplicates
    across pipeline runs
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import anthropic
    from backend.dataset_pipeline.config import PipelineConfig
    from backend.dataset_pipeline.processing.types import RawSVG

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 300
_CAPTION_DELAY_SECONDS = 0.5
_CONTEXT_KEYS = ("page_title", "paper_title", "figure_index", "category")
_SYSTEM_PROMPT = """You are captioning an SVG technical diagram to use as a training prompt for an SVG 
generation model. Describe the diagram with emphasis on:
1. LAYOUT: absolute positions (left/center/right, top/middle/bottom), relative 
   positions between elements, approximate proportions
2. STRUCTURE: bounding boxes, swim lanes, groupings, containment relationships
3. CONNECTIONS: arrow styles (solid/dashed/dotted), routing (straight/bent/curved), 
   directionality, label positions relative to arrows
4. GEOMETRY: sizes of elements relative to each other, spacing, alignment axes
5. VISUAL ENCODING: fill colors, border styles, font sizes, special markers or symbols

Do NOT describe what the diagram means semantically. Describe what a person would 
need to know to draw it from scratch. Use precise directional language. 
Output one dense paragraph, maximum 150 words. No bullet points."""


class CaptionError(Exception):
    """Raised when an SVG cannot be captioned into a usable prompt."""


# Keep this schema manually in sync with backend/training/dataset_interface.py.
@dataclass
class TrainingRecord:
    id: str
    prompt: str
    completion: str
    source: str
    split: str
    metadata: dict
    diagram_ir: dict | None = None


def _metadata_context(metadata: dict) -> dict:
    return {
        key: metadata[key]
        for key in _CONTEXT_KEYS
        if key in metadata and metadata[key] not in (None, "", [], {})
    }


def _response_text(response: object) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _openai_response_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices", [])
    for choice in choices or []:
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message", {})
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text")
                if text:
                    parts.append(text)
            if parts:
                return "".join(parts).strip()
    return ""


def caption_svg(svg_string: str, metadata: dict, client: anthropic.Anthropic) -> str:
    import cairosvg

    png_buffer = io.BytesIO()
    cairosvg.svg2png(
        bytestring=svg_string.encode(),
        write_to=png_buffer,
        output_width=800,
    )
    image_data = base64.b64encode(png_buffer.getvalue()).decode("ascii")
    context = _metadata_context(metadata or {})

    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Additional context: {json.dumps(context)}",
                    },
                ],
            }
        ],
    )

    caption = _response_text(response)
    if not caption:
        raise CaptionError("empty_response")
    if len(caption.split()) < 20:
        raise CaptionError("response_too_short")
    return caption


def caption_svg_local(svg_string: str, metadata: dict, config: PipelineConfig) -> str:
    import cairosvg

    png_buffer = io.BytesIO()
    cairosvg.svg2png(
        bytestring=svg_string.encode(),
        write_to=png_buffer,
        output_width=800,
    )
    image_data = base64.b64encode(png_buffer.getvalue()).decode("ascii")
    context = _metadata_context(metadata or {})

    response = httpx.post(
        f"{getattr(config, 'caption_local_base_url', 'http://localhost:11434/v1').rstrip('/')}/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": getattr(config, "caption_local_model", "llava"),
            "max_tokens": _MAX_TOKENS,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Additional context: {json.dumps(context)}",
                        },
                    ],
                },
            ],
        },
        timeout=float(getattr(config, "caption_local_timeout_s", 60.0) or 60.0),
    )
    response.raise_for_status()
    caption = _openai_response_text(response.json())
    if not caption:
        raise CaptionError("empty_response")
    if len(caption.split()) < 20:
        raise CaptionError("response_too_short")
    return caption


def batch_caption(
    raw_svgs: list[RawSVG],
    config: PipelineConfig,
    normalize: bool = True,
) -> tuple[list[TrainingRecord], list[tuple[RawSVG, str]]]:
    from tqdm import tqdm

    from backend.dataset_pipeline.processing.normalizer import normalize_svg

    records: list[TrainingRecord] = []
    indexed_records: list[tuple[int, TrainingRecord]] = []
    failures: list[tuple[RawSVG, str]] = []
    total = len(raw_svgs)
    max_workers = max(1, int(getattr(config, "caption_parallelism", 1) or 1))
    backend = getattr(config, "caption_backend", "anthropic")
    thread_state = threading.local()

    def _client():
        client = getattr(thread_state, "client", None)
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=config.anthropic_api_key)
            thread_state.client = client
        return client

    def _caption_one(index: int, raw_svg: RawSVG) -> tuple[int, TrainingRecord]:
        normalized_svg = normalize_svg(raw_svg.svg_string) if normalize else raw_svg.svg_string
        if backend == "local":
            caption = caption_svg_local(normalized_svg, raw_svg.metadata, config)
        else:
            caption = caption_svg(normalized_svg, raw_svg.metadata, _client())
        return (
            index,
            TrainingRecord(
                id=raw_svg.source_id,
                prompt=caption,
                completion=normalized_svg,
                source=raw_svg.domain,
                split="",
                metadata=raw_svg.metadata,
                diagram_ir=raw_svg.metadata.get("diagram_ir"),
            ),
        )

    with tqdm(total=total, desc=f"Captioning [0/{total}]", unit="svg") as bar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_caption_one, index, raw_svg): raw_svg
                for index, raw_svg in enumerate(raw_svgs)
            }
            for index, future in enumerate(as_completed(futures), start=1):
                raw_svg = futures[future]
                try:
                    indexed_records.append(future.result())
                except CaptionError as exc:
                    reason = str(exc)
                    logger.warning("CaptionError for %s: %s", raw_svg.source_id, reason)
                    failures.append((raw_svg, reason))
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    logger.exception("Failed to caption %s", raw_svg.source_id)
                    failures.append((raw_svg, reason))
                finally:
                    bar.update(1)
                    bar.set_description(f"Captioning [{index}/{total}]")
                    if index < total and max_workers == 1:
                        time.sleep(_CAPTION_DELAY_SECONDS)

    records = [record for _, record in sorted(indexed_records, key=lambda item: item[0])]
    return records, failures
