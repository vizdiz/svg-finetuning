from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import time
from typing import Any, Callable, Iterable

import httpx

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.corpus.schema import (
    CorpusCandidate,
    build_manifest,
    read_candidates,
    utc_now,
    write_jsonl,
)
from backend.dataset_pipeline.processing.validator import (
    SVGValidationResult,
    svg_quality_score,
    validate_svg_detailed,
)


FetchFn = Callable[[CorpusCandidate], str]


@dataclass(slots=True)
class FetchValidateStats:
    total: int = 0
    skipped_existing: int = 0
    fetched: int = 0
    rejected: int = 0
    failed: int = 0
    limited: bool = False
    reasons: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "skipped_existing": self.skipped_existing,
            "fetched": self.fetched,
            "rejected": self.rejected,
            "failed": self.failed,
            "limited": self.limited,
            "reasons": dict(sorted(self.reasons.items())),
        }


@dataclass(frozen=True, slots=True)
class FetchValidateResult:
    corpus_id: str
    candidates: str
    manifest: str
    summary: str
    assets_dir: str
    stats: FetchValidateStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "candidates": self.candidates,
            "manifest": self.manifest,
            "summary": self.summary,
            "assets_dir": self.assets_dir,
            "stats": self.stats.to_dict(),
        }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _asset_name(candidate: CorpusCandidate, svg_string: str) -> str:
    content_hash = _sha256_text(svg_string)[:20]
    return f"{candidate.candidate_id}-{content_hash}.svg"


def _fetch_with_retry(
    candidate: CorpusCandidate,
    *,
    client: httpx.Client,
    retry_delays: Iterable[float],
) -> str:
    last_exc: Exception | None = None
    for attempt, delay in enumerate([*retry_delays, 0], start=1):
        try:
            response = client.get(candidate.uri, timeout=60)
            response.raise_for_status()
            return response.text
        except Exception as exc:  # pragma: no cover - exercised through injected fetcher in unit tests
            last_exc = exc
            if delay <= 0:
                break
            time.sleep(delay)
    raise RuntimeError(f"fetch_failed:{type(last_exc).__name__}: {last_exc}") from last_exc


def _status_candidate(
    candidate: CorpusCandidate,
    *,
    status: str,
    metadata: dict[str, Any],
) -> CorpusCandidate:
    payload = candidate.to_dict()
    payload["fetch_status"] = status
    payload["metadata"] = {**candidate.metadata, **metadata}
    return CorpusCandidate.from_dict(payload)


def _fetched_candidate(
    candidate: CorpusCandidate,
    *,
    svg_string: str,
    asset_path: Path,
    validation: SVGValidationResult,
    s3_uri: str = "",
) -> CorpusCandidate:
    payload = candidate.to_dict()
    payload["fetch_status"] = "fetched"
    payload["content_sha256"] = _sha256_text(svg_string)
    payload["local_path"] = str(asset_path)
    payload["s3_uri"] = s3_uri
    payload["metadata"] = {
        **candidate.metadata,
        "validation": {
            "ok": True,
            "warnings": validation.warnings,
            "stats": validation.stats,
            "quality_score": svg_quality_score(validation),
        },
        "fetched_at": utc_now(),
    }
    return CorpusCandidate.from_dict(payload)


def _write_summary(path: Path, corpus_id: str, stats: FetchValidateStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "corpus_id": corpus_id,
                "created_at": utc_now(),
                "stats": stats.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


def fetch_validate_candidates(
    *,
    corpus_id: str,
    input_candidates: Path,
    output_root: Path,
    config: ScrapeConfig | None = None,
    limit: int | None = None,
    fetch_fn: FetchFn | None = None,
    retry_delays: Iterable[float] = (1, 2, 4),
    asset_s3_prefix: str = "",
    request_delay_s: float = 0.0,
) -> FetchValidateResult:
    scrape_config = config or ScrapeConfig()
    candidates = read_candidates(input_candidates)
    stats = FetchValidateStats(total=len(candidates))

    corpus_root = output_root / corpus_id
    assets_dir = corpus_root / "assets" / "wikimedia"
    output_candidates = corpus_root / "candidates.jsonl"
    manifest_path = corpus_root / "manifest.json"
    summary_path = corpus_root / "fetch_validation_summary.json"

    processed = 0
    updated: list[CorpusCandidate] = []

    with httpx.Client(headers={"User-Agent": "svg-finetuning-corpus-fetcher/1.0"}) as client:
        for candidate in candidates:
            if candidate.fetch_status == "fetched":
                stats.skipped_existing += 1
                updated.append(candidate)
                continue
            if limit is not None and processed >= limit:
                stats.limited = True
                updated.append(candidate)
                continue
            if candidate.source != "wikimedia":
                updated.append(candidate)
                continue

            processed += 1
            try:
                svg_string = (
                    fetch_fn(candidate)
                    if fetch_fn is not None
                    else _fetch_with_retry(candidate, client=client, retry_delays=retry_delays)
                )
                if request_delay_s > 0 and fetch_fn is None:
                    time.sleep(request_delay_s)
            except Exception as exc:
                stats.failed += 1
                reason = f"fetch_failed:{type(exc).__name__}"
                stats.reasons[reason] += 1
                updated.append(
                    _status_candidate(
                        candidate,
                        status="failed",
                        metadata={"fetch_error": str(exc), "fetch_error_type": type(exc).__name__},
                    )
                )
                continue

            validation = validate_svg_detailed(svg_string, scrape_config)
            if not validation.ok:
                stats.rejected += 1
                reason = validation.reason
                stats.reasons[reason] += 1
                updated.append(
                    _status_candidate(
                        candidate,
                        status="rejected",
                        metadata={
                            "validation": {
                                "ok": False,
                                "hard_reject_reasons": validation.hard_reject_reasons,
                                "warnings": validation.warnings,
                                "stats": validation.stats,
                            },
                            "fetched_at": utc_now(),
                        },
                    )
                )
                continue

            assets_dir.mkdir(parents=True, exist_ok=True)
            asset_path = assets_dir / _asset_name(candidate, svg_string)
            asset_path.write_text(svg_string)
            s3_uri = f"{asset_s3_prefix.rstrip('/')}/{asset_path.name}" if asset_s3_prefix else ""
            stats.fetched += 1
            updated.append(
                _fetched_candidate(
                    candidate,
                    svg_string=svg_string,
                    asset_path=asset_path,
                    validation=validation,
                    s3_uri=s3_uri,
                )
            )

    write_jsonl(output_candidates, [candidate.to_dict() for candidate in updated])
    manifest = build_manifest(
        corpus_id,
        str(output_candidates),
        updated,
        metadata={
            "source": "wikimedia",
            "stage": "fetch_validate",
            "input_candidates": str(input_candidates),
            "assets_dir": str(assets_dir),
            "stats": stats.to_dict(),
        },
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    _write_summary(summary_path, corpus_id, stats)

    return FetchValidateResult(
        corpus_id=corpus_id,
        candidates=str(output_candidates),
        manifest=str(manifest_path),
        summary=str(summary_path),
        assets_dir=str(assets_dir),
        stats=stats,
    )
