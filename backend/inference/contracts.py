"""
backend.inference.contracts

Internal contract between the public API and the SageMaker-hosted vLLM
endpoint.

The API layer owns:
  - sessions
  - cache keys
  - branching / revision lineage

The model endpoint owns:
  - SVG generation from the request payload

This contract is deliberately narrow and JSON-serializable so it can be
used across Lambda, SageMaker, local tests, and any future endpoint host.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _non_empty(value: str | None) -> str:
    return value or ""


@dataclass
class ModelGenerationRequest:
    request_id: str
    session_id: str = ""
    branch_id: str = ""
    revision_index: int = 0
    parent_request_id: str = ""
    prompt: str = ""
    feedback: str = ""
    feedback_rating: int | None = None
    model: str = ""
    system_prompt: str = "You generate SVG only."
    style: str = "technical"
    temperature: float = 0.0
    width: int = 640
    height: int = 480
    seed: int | None = None
    cache_namespace: str = "default"
    max_tokens: int = 512
    input_mode: str = "text"
    reference_images: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)
    media_metadata: dict[str, Any] = field(default_factory=dict)
    diagram_description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.request_id:
            raise ValueError("request_id is required")
        has_media = bool(self.reference_images or self.attachments or self.diagram_description.strip())
        if not self.prompt.strip() and not has_media:
            raise ValueError("prompt or diagram description is required")
        if int(self.revision_index) < 0:
            raise ValueError("revision_index must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.input_mode and self.input_mode not in {"text", "image", "mixed"}:
            raise ValueError("input_mode must be text, image, or mixed")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelGenerationRequest":
        request = cls(
            request_id=str(payload.get("request_id", "")),
            session_id=str(payload.get("session_id", "")),
            branch_id=str(payload.get("branch_id", "")),
            revision_index=int(payload.get("revision_index", 0)),
            parent_request_id=str(payload.get("parent_request_id", "")),
            prompt=str(payload.get("prompt", "")),
            feedback=str(payload.get("feedback", "")),
            feedback_rating=payload.get("feedback_rating"),
            model=str(payload.get("model", "")),
            system_prompt=str(payload.get("system_prompt", "You generate SVG only.")),
            style=str(payload.get("style", "technical")),
            temperature=float(payload.get("temperature", 0.0)),
            width=int(payload.get("width", 640)),
            height=int(payload.get("height", 480)),
            seed=payload.get("seed"),
            cache_namespace=str(payload.get("cache_namespace", "default")),
            max_tokens=int(payload.get("max_tokens", 512)),
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
class ModelGenerationResponse:
    request_id: str
    svg: str
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
