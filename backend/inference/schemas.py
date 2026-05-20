"""
inference.schemas

Explicit request/response schemas for the cache-first generation API.

These are plain dataclasses so they stay lightweight in Lambda, ECS, and
local unit tests without pulling in a validation framework.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import json


def _non_empty(value: str | None) -> str:
    return value or ""


@dataclass
class GenerateRequest:
    prompt: str
    request_id: str = ""
    session_id: str = ""
    parent_request_id: str = ""
    branch_id: str = ""
    revision_index: int = 0
    feedback: str = ""
    feedback_rating: int | None = None
    model: str = ""
    style: str = ""
    temperature: float = 0.0
    width: int = 640
    height: int = 480
    seed: int | None = None
    cache_namespace: str = "default"
    max_tokens: int = 2048
    input_mode: str = "text"
    reference_images: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)
    media_metadata: dict[str, Any] = field(default_factory=dict)
    diagram_description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        has_media = bool(self.reference_images or self.attachments or self.diagram_description.strip())
        if not self.prompt.strip() and not has_media:
            raise ValueError("prompt or media input is required")
        if int(self.revision_index) < 0:
            raise ValueError("revision_index must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.input_mode and self.input_mode not in {"text", "image", "mixed"}:
            raise ValueError("input_mode must be text, image, or mixed")

    def cache_payload(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt.strip(),
            "model": _non_empty(self.model).strip(),
            "style": _non_empty(self.style).strip(),
            "temperature": self.temperature,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "cache_namespace": _non_empty(self.cache_namespace).strip(),
            "max_tokens": self.max_tokens,
            "session_id": _non_empty(self.session_id).strip(),
            "parent_request_id": _non_empty(self.parent_request_id).strip(),
            "branch_id": _non_empty(self.branch_id).strip(),
            "revision_index": int(self.revision_index),
            "feedback": _non_empty(self.feedback).strip(),
            "feedback_rating": self.feedback_rating,
            "input_mode": _non_empty(self.input_mode).strip(),
            "reference_images": list(self.reference_images or []),
            "attachments": list(self.attachments or []),
            "media_metadata": self.media_metadata,
            "diagram_description": self.diagram_description.strip(),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerateRequest":
        request = cls(
            request_id=str(payload.get("request_id", "")),
            prompt=str(payload.get("prompt", "")),
            session_id=str(payload.get("session_id", "")),
            parent_request_id=str(payload.get("parent_request_id", "")),
            branch_id=str(payload.get("branch_id", "")),
            revision_index=int(payload.get("revision_index", 0)),
            feedback=str(payload.get("feedback", "")),
            feedback_rating=payload.get("feedback_rating"),
            model=str(payload.get("model", "")),
            style=str(payload.get("style", "")),
            temperature=float(payload.get("temperature", 0.0)),
            width=int(payload.get("width", 640)),
            height=int(payload.get("height", 480)),
            seed=payload.get("seed"),
            cache_namespace=str(payload.get("cache_namespace", "default")),
            max_tokens=int(payload.get("max_tokens", 2048)),
            input_mode=str(payload.get("input_mode", "text")),
            reference_images=list(payload.get("reference_images", []) or []),
            attachments=list(payload.get("attachments", []) or []),
            media_metadata=dict(payload.get("media_metadata", {}) or {}),
            diagram_description=str(payload.get("diagram_description", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )
        request.validate()
        return request


@dataclass
class GenerateResponse:
    request_id: str
    cache_key: str
    cached: bool
    svg: str
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UploadRequest:
    request_id: str = ""
    filename: str = ""
    svg: str = ""
    note: str = ""
    source: str = "web"
    model_name: str = ""

    def validate(self) -> None:
        if not self.svg.strip():
            raise ValueError("svg is required")
        if not self.filename.strip():
            raise ValueError("filename is required")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UploadRequest":
        request = cls(
            request_id=str(payload.get("request_id", "")),
            filename=str(payload.get("filename", "")),
            svg=str(payload.get("svg", "")),
            note=str(payload.get("note", "")),
            source=str(payload.get("source", "web")),
            model_name=str(payload.get("model_name", "")),
        )
        request.validate()
        return request


@dataclass
class UploadResponse:
    request_id: str
    accepted: bool
    asset_key: str
    bytes: int
    download_allowed: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ApiErrorResponse:
    error: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def encode_response(payload: GenerateResponse | UploadResponse | ApiErrorResponse) -> str:
    return json.dumps(payload.to_dict())
