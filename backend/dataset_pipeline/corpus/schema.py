from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
CORPUS_ROOT_PREFIX = "corpus"

VALID_SOURCES = {
    "synthetic",
    "arxiv",
    "wikimedia",
    "github",
    "commoncrawl",
}

VALID_TARGET_FORMATS = {
    "diagram_ir",
    "raw_svg",
    "svg_to_ir",
}

VALID_FETCH_STATUSES = {
    "not_required",
    "pending",
    "fetched",
    "failed",
    "rejected",
}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(*parts: object, prefix: str = "candidate") -> str:
    raw = "\n".join(str(part) for part in parts).encode()
    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:20]}"


@dataclass(slots=True)
class CorpusCandidate:
    candidate_id: str
    source: str
    uri: str
    target_format: str
    title: str = ""
    caption: str = ""
    license: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    priority_score: float = 0.0
    fetch_status: str = "pending"
    content_sha256: str = ""
    local_path: str = ""
    s3_uri: str = ""
    created_at: str = field(default_factory=utc_now)

    def validate(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if self.source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}, got {self.source!r}")
        if not self.uri:
            raise ValueError("uri is required")
        if self.target_format not in VALID_TARGET_FORMATS:
            raise ValueError(
                f"target_format must be one of {sorted(VALID_TARGET_FORMATS)}, got {self.target_format!r}"
            )
        if self.fetch_status not in VALID_FETCH_STATUSES:
            raise ValueError(
                f"fetch_status must be one of {sorted(VALID_FETCH_STATUSES)}, got {self.fetch_status!r}"
            )
        if not 0.0 <= float(self.priority_score) <= 1.0:
            raise ValueError("priority_score must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CorpusCandidate":
        candidate = cls(
            candidate_id=str(payload.get("candidate_id", "")),
            source=str(payload.get("source", "")),
            uri=str(payload.get("uri", "")),
            target_format=str(payload.get("target_format", "")),
            title=str(payload.get("title", "")),
            caption=str(payload.get("caption", "")),
            license=str(payload.get("license", "")),
            provenance=dict(payload.get("provenance", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
            priority_score=float(payload.get("priority_score", 0.0)),
            fetch_status=str(payload.get("fetch_status", "pending")),
            content_sha256=str(payload.get("content_sha256", "")),
            local_path=str(payload.get("local_path", "")),
            s3_uri=str(payload.get("s3_uri", "")),
            created_at=str(payload.get("created_at", utc_now())),
        )
        candidate.validate()
        return candidate


@dataclass(slots=True)
class CorpusManifest:
    corpus_id: str
    created_at: str
    candidates_uri: str
    record_count: int
    schema_version: str = SCHEMA_VERSION
    source_counts: dict[str, int] = field(default_factory=dict)
    target_format_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if not self.corpus_id:
            raise ValueError("corpus_id is required")
        if not self.candidates_uri:
            raise ValueError("candidates_uri is required")
        if self.record_count < 0:
            raise ValueError("record_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_candidates(path: Path) -> list[CorpusCandidate]:
    records: list[CorpusCandidate] = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(CorpusCandidate.from_dict(json.loads(line)))
    return records


def build_manifest(corpus_id: str, candidates_uri: str, candidates: list[CorpusCandidate], metadata: dict[str, Any] | None = None) -> CorpusManifest:
    source_counts: dict[str, int] = {}
    target_format_counts: dict[str, int] = {}
    for candidate in candidates:
        candidate.validate()
        source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
        target_format_counts[candidate.target_format] = target_format_counts.get(candidate.target_format, 0) + 1
    return CorpusManifest(
        corpus_id=corpus_id,
        created_at=utc_now(),
        candidates_uri=candidates_uri,
        record_count=len(candidates),
        source_counts=source_counts,
        target_format_counts=target_format_counts,
        metadata=metadata or {},
    )


def local_corpus_paths(output_root: Path, corpus_id: str) -> dict[str, Path]:
    root = output_root / corpus_id
    return {
        "root": root,
        "candidates": root / "candidates.jsonl",
        "manifest": root / "manifest.json",
        "training": root / "training",
        "eval": root / "eval",
        "assets": root / "assets",
    }


def s3_corpus_prefix(corpus_id: str) -> str:
    return f"{CORPUS_ROOT_PREFIX}/{corpus_id}"
