"""
backend.inference.sessions

Session persistence for iterative prompting.

The session store is intentionally simple:
  - one record per session_id
  - each record keeps the prompt, generation history, and feedback history
  - the persistence backend can be local JSON files, DynamoDB, or S3

This is enough for branching, revision history, and download continuity
without forcing the UI to manage state across page reloads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
import json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class SessionGeneration:
    request_id: str
    cache_key: str
    branch_id: str
    revision_index: int
    parent_request_id: str = ""
    prompt: str = ""
    feedback: str = ""
    feedback_rating: int | None = None
    model: str = ""
    cached: bool = False
    svg: str = ""
    created_at: str = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionFeedback:
    request_id: str
    rating: int
    prompt: str
    response_svg: str
    created_at: str
    session_id: str = ""
    branch_id: str = ""
    revision_index: int = 0
    source: str = "web"
    model_name: str = ""
    comment: str = ""
    download_allowed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionRecord:
    session_id: str
    prompt: str = ""
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    current_request_id: str = ""
    current_branch_id: str = ""
    current_revision_index: int = 0
    current_svg: str = ""
    current_cache_key: str = ""
    generations: list[SessionGeneration] = field(default_factory=list)
    feedback: list[SessionFeedback] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = _utc_now()

    def append_generation(self, generation: SessionGeneration) -> None:
        if not self.prompt:
            self.prompt = generation.prompt
        self.current_request_id = generation.request_id
        self.current_branch_id = generation.branch_id
        self.current_revision_index = generation.revision_index
        self.current_svg = generation.svg
        self.current_cache_key = generation.cache_key
        self.generations.append(generation)
        self.metadata.update(
            {
                "last_feedback": generation.feedback,
                "last_feedback_rating": generation.feedback_rating,
                "last_model": generation.model,
            }
        )
        self.touch()

    def append_feedback(self, record: SessionFeedback) -> None:
        self.feedback.append(record)
        self.metadata.update(
            {
                "last_feedback_request_id": record.request_id,
                "last_feedback_rating": record.rating,
                "last_download_allowed": record.download_allowed,
            }
        )
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_request_id": self.current_request_id,
            "current_branch_id": self.current_branch_id,
            "current_revision_index": self.current_revision_index,
            "current_svg": self.current_svg,
            "current_cache_key": self.current_cache_key,
            "generations": [asdict(item) for item in self.generations],
            "feedback": [asdict(item) for item in self.feedback],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionRecord":
        record = cls(
            session_id=str(payload.get("session_id", "")),
            prompt=str(payload.get("prompt", "")),
            created_at=str(payload.get("created_at", _utc_now())),
            updated_at=str(payload.get("updated_at", _utc_now())),
            current_request_id=str(payload.get("current_request_id", "")),
            current_branch_id=str(payload.get("current_branch_id", "")),
            current_revision_index=int(payload.get("current_revision_index", 0)),
            current_svg=str(payload.get("current_svg", "")),
            current_cache_key=str(payload.get("current_cache_key", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )
        for item in payload.get("generations", []) or []:
            record.generations.append(SessionGeneration(**item))
        for item in payload.get("feedback", []) or []:
            record.feedback.append(SessionFeedback(**item))
        return record


class SessionStore(Protocol):
    def load(self, session_id: str) -> SessionRecord | None: ...

    def save(self, record: SessionRecord) -> None: ...


class InMemorySessionStore:
    def __init__(self):
        self._records: dict[str, SessionRecord] = {}

    def load(self, session_id: str) -> SessionRecord | None:
        return self._records.get(session_id)

    def save(self, record: SessionRecord) -> None:
        self._records[record.session_id] = record


class LocalJsonSessionStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def load(self, session_id: str) -> SessionRecord | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        return SessionRecord.from_dict(json.loads(path.read_text()))

    def save(self, record: SessionRecord) -> None:
        path = self._path(record.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True))


class DynamoDBSessionStore:
    def __init__(self, table_name: str, *, region: str = "us-east-1", ddb=None):
        import boto3

        self.table_name = table_name
        self.ddb = ddb or boto3.resource("dynamodb", region_name=region)
        self.table = self.ddb.Table(table_name)

    def load(self, session_id: str) -> SessionRecord | None:
        response = self.table.get_item(Key={"session_id": session_id})
        item = response.get("Item")
        if not item:
            return None
        return SessionRecord.from_dict(item)

    def save(self, record: SessionRecord) -> None:
        self.table.put_item(Item=record.to_dict())
