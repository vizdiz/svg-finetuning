"""
feedback.schema

JSONL feedback records are intentionally compact and stable so they can be
stored in S3, DynamoDB, or a relational store without translation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import json


@dataclass
class FeedbackRecord:
    request_id: str
    rating: int
    created_at: str
    prompt: str
    response_svg: str
    session_id: str = ""
    branch_id: str = ""
    revision_index: int = 0
    source: str = "web"
    model_name: str = ""
    download_allowed: bool = False
    comment: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.request_id:
            raise ValueError("request_id is required")
        if not (1 <= int(self.rating) <= 5):
            raise ValueError("rating must be between 1 and 5")
        if int(self.revision_index) < 0:
            raise ValueError("revision_index must be non-negative")
        if not self.prompt:
            raise ValueError("prompt is required")
        if not self.response_svg:
            raise ValueError("response_svg is required")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeedbackRecord":
        record = cls(
            request_id=str(payload.get("request_id", "")),
            rating=int(payload.get("rating", 0)),
            created_at=str(payload.get("created_at") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
            prompt=str(payload.get("prompt", "")),
            response_svg=str(payload.get("response_svg", "")),
            session_id=str(payload.get("session_id", "")),
            branch_id=str(payload.get("branch_id", "")),
            revision_index=int(payload.get("revision_index", 0)),
            source=str(payload.get("source", "web")),
            model_name=str(payload.get("model_name", "")),
            download_allowed=bool(payload.get("download_allowed", False)),
            comment=str(payload.get("comment", "")),
            metadata=dict(payload.get("metadata", {})),
        )
        record.validate()
        return record


@dataclass
class FeedbackIngestRequest:
    request_id: str
    rating: int
    prompt: str
    response_svg: str
    created_at: str = ""
    session_id: str = ""
    branch_id: str = ""
    revision_index: int = 0
    source: str = "web"
    model_name: str = ""
    comment: str = ""
    download_allowed: bool = False
    cache_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.request_id:
            raise ValueError("request_id is required")
        if not (1 <= int(self.rating) <= 5):
            raise ValueError("rating must be between 1 and 5")
        if int(self.revision_index) < 0:
            raise ValueError("revision_index must be non-negative")
        if not self.prompt:
            raise ValueError("prompt is required")
        if not self.response_svg:
            raise ValueError("response_svg is required")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeedbackIngestRequest":
        record = cls(
            request_id=str(payload.get("request_id", "")),
            rating=int(payload.get("rating", 0)),
            prompt=str(payload.get("prompt", "")),
            response_svg=str(payload.get("response_svg", "")),
            created_at=str(payload.get("created_at", "")),
            session_id=str(payload.get("session_id", "")),
            branch_id=str(payload.get("branch_id", "")),
            revision_index=int(payload.get("revision_index", 0)),
            source=str(payload.get("source", "web")),
            model_name=str(payload.get("model_name", "")),
            comment=str(payload.get("comment", "")),
            download_allowed=bool(payload.get("download_allowed", False)),
            cache_key=str(payload.get("cache_key", "")),
            metadata=dict(payload.get("metadata", {})),
        )
        record.validate()
        return record


@dataclass
class FeedbackIngestResponse:
    request_id: str
    accepted: bool
    download_allowed: bool
    feedback_uri: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
