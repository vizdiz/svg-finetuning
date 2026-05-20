"""
backend.inference.multimodal

Multimodal request normalization for the generation API.

Supported flow:
  - text input -> polished diagram brief -> text model
  - image/media input -> vision description -> polished diagram brief -> text model

The normalization step is intentionally separate from the generator so we can
swap in a real vision model later without changing the public API contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from backend.inference.schemas import GenerateRequest


class MediaDescriptionClient(Protocol):
    def describe(self, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class MultimodalNormalizationResult:
    original_prompt: str
    normalized_prompt: str
    diagram_description: str
    media_summary: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _flatten_media_metadata(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _summarize_media_item(uri: str, metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    filename = str(metadata.get("filename") or metadata.get("name") or uri.rsplit("/", 1)[-1])
    if filename:
        parts.append(filename)
    mime_type = str(metadata.get("mime_type") or metadata.get("mime") or "")
    if mime_type:
        parts.append(mime_type)
    width = metadata.get("width")
    height = metadata.get("height")
    if width and height:
        parts.append(f"{width}x{height}")
    alt_text = str(metadata.get("alt_text") or metadata.get("alt") or "").strip()
    caption = str(metadata.get("caption") or metadata.get("description") or "").strip()
    text_bits = [bit for bit in (alt_text, caption) if bit]
    if text_bits:
        parts.append(" - ".join(text_bits))
    return "; ".join(parts)


def _normalize_text_prompt(prompt: str) -> str:
    prompt = " ".join(prompt.strip().split())
    if not prompt:
        return ""
    return (
        "Diagram brief:\n"
        f"{prompt}\n\n"
        "Preserve the user's intent, entities, relationships, layout constraints, and labels. "
        "Do not invent extra content. Return SVG only."
    )


def _build_media_description(
    request: GenerateRequest,
    *,
    description_client: MediaDescriptionClient | None = None,
) -> tuple[str, list[str], dict[str, Any]]:
    media_items = list(request.reference_images or []) + list(request.attachments or [])
    media_metadata = _flatten_media_metadata(request.media_metadata)
    payload = request.to_dict()
    payload["media_metadata"] = media_metadata

    if description_client is not None and media_items:
        result = description_client.describe(payload)
        diagram_description = str(result.get("diagram_description", "")).strip()
        media_summary = [str(item).strip() for item in result.get("media_summary", []) if str(item).strip()]
        metadata = dict(result.get("metadata", {}) or {})
        if not media_summary:
            media_summary = [_summarize_media_item(str(uri), _flatten_media_metadata(media_metadata.get(str(uri), {}))) for uri in media_items]
        return diagram_description, media_summary, metadata

    summaries: list[str] = []
    for uri in media_items:
        summaries.append(_summarize_media_item(str(uri), _flatten_media_metadata(media_metadata.get(str(uri), {}))))

    if summaries:
        diagram_description = "Reference media summary:\n" + "\n".join(f"- {line}" for line in summaries)
    elif request.diagram_description.strip():
        diagram_description = request.diagram_description.strip()
    else:
        diagram_description = ""
    return diagram_description, summaries, {}


def build_diagram_description(
    request: GenerateRequest,
    *,
    description_client: MediaDescriptionClient | None = None,
) -> MultimodalNormalizationResult:
    original_prompt = request.prompt.strip()
    diagram_description, media_summary, media_metadata = _build_media_description(
        request,
        description_client=description_client,
    )

    prompt_parts: list[str] = []
    if original_prompt:
        prompt_parts.append(
            "User request:\n"
            f"{original_prompt}\n"
        )
    if diagram_description:
        prompt_parts.append(
            "Diagram description:\n"
            f"{diagram_description}\n"
        )
    if media_summary and not diagram_description:
        prompt_parts.append(
            "Reference media summary:\n"
            + "\n".join(f"- {line}" for line in media_summary)
            + "\n"
        )
    prompt_parts.append(
        "Task:\n"
        "Generate a clean SVG technical diagram that follows the brief above. "
        "Preserve entities, relationships, labels, and layout constraints. "
        "Do not invent unsupported content. Return SVG only."
    )
    normalized_prompt = "\n".join(part.strip() for part in prompt_parts if part.strip())

    metadata = {
        "input_mode": request.input_mode,
        "reference_images": list(request.reference_images or []),
        "attachments": list(request.attachments or []),
    }
    if media_metadata:
        metadata["media_metadata"] = media_metadata
    if diagram_description:
        metadata["diagram_description"] = diagram_description
    if original_prompt:
        metadata["original_prompt"] = original_prompt
    if normalized_prompt:
        metadata["normalized_prompt"] = normalized_prompt

    return MultimodalNormalizationResult(
        original_prompt=original_prompt,
        normalized_prompt=normalized_prompt,
        diagram_description=diagram_description,
        media_summary=media_summary,
        metadata=metadata,
    )


def normalize_generate_request(
    request: GenerateRequest,
    *,
    description_client: MediaDescriptionClient | None = None,
) -> MultimodalNormalizationResult:
    result = build_diagram_description(request, description_client=description_client)
    if result.normalized_prompt or request.input_mode in {"image", "mixed"}:
        request.prompt = result.normalized_prompt
        request.diagram_description = result.diagram_description
        if result.media_summary:
            request.input_mode = "mixed" if result.original_prompt else "image"
        request.metadata = {**getattr(request, "metadata", {}), **result.metadata}
    return result
