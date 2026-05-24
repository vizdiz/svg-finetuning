from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class WorkerResult:
    corpus_id: str
    candidates: Path
    manifest: Path
    records: int


class SourceWorker(Protocol):
    source: str

    def discover(self, *, corpus_id: str, output_root: Path, limit: int) -> WorkerResult:
        """Write a normalized candidate manifest without fetching origin bytes."""

    def fetch_extract(self, *, candidates: Path, corpus_id: str, output_root: Path, limit: int | None = None) -> WorkerResult:
        """Fetch or extract selected source bytes and write updated candidate state."""
