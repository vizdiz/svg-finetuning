from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any, Callable

import httpx

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.corpus.schema import CorpusCandidate, build_manifest, local_corpus_paths, read_candidates, utc_now, write_jsonl
from backend.dataset_pipeline.corpus.s3_orchestration import read_commoncrawl_warc_record
from backend.dataset_pipeline.processing.validator import validate_svg_detailed


FetchText = Callable[[CorpusCandidate], str]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _asset_path(root: Path, candidate: CorpusCandidate, svg_string: str) -> Path:
    name = f"{candidate.candidate_id}-{_sha256(svg_string)[:20]}.svg"
    return root / candidate.source / name


def _http_fetch(candidate: CorpusCandidate) -> str:
    if candidate.source == "commoncrawl":
        raise RuntimeError("commoncrawl candidates must be read from WARC/WAT records, not fetched from origin")
    with httpx.Client(headers={"User-Agent": "svg-finetuning-indexed-fetcher/1.0"}) as client:
        response = client.get(candidate.uri, timeout=60)
        response.raise_for_status()
        return response.text


def _local_record_fetch(candidate: CorpusCandidate) -> str:
    record_path = candidate.metadata.get("record_path") or candidate.metadata.get("local_path")
    if not record_path:
        if candidate.source == "commoncrawl":
            return read_commoncrawl_warc_record(candidate)
        return _http_fetch(candidate)
    return Path(str(record_path)).read_text()


def fetch_indexed_svg_candidates(
    *,
    corpus_id: str,
    input_candidates: Path,
    output_root: Path,
    config: ScrapeConfig | None = None,
    limit: int | None = None,
    fetch_fn: FetchText | None = None,
) -> dict[str, Any]:
    scrape_config = config or ScrapeConfig()
    source_candidates = read_candidates(input_candidates)
    paths = local_corpus_paths(output_root, corpus_id)
    assets_dir = paths["assets"]
    assets_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(source_candidates), "fetched": 0, "rejected": 0, "failed": 0, "limited": False}
    output: list[CorpusCandidate] = []
    processed = 0

    for candidate in source_candidates:
        if candidate.source not in {"github", "commoncrawl"}:
            output.append(candidate)
            continue
        if limit is not None and processed >= limit:
            stats["limited"] = True
            output.append(candidate)
            continue
        processed += 1
        try:
            svg_string = (fetch_fn or _local_record_fetch)(candidate)
        except Exception as exc:
            stats["failed"] += 1
            payload = candidate.to_dict()
            payload["fetch_status"] = "failed"
            payload["metadata"] = {**candidate.metadata, "fetch_error": str(exc), "fetch_error_type": type(exc).__name__}
            output.append(CorpusCandidate.from_dict(payload))
            continue

        validation = validate_svg_detailed(svg_string, scrape_config)
        if not validation.ok:
            stats["rejected"] += 1
            payload = candidate.to_dict()
            payload["fetch_status"] = "rejected"
            payload["metadata"] = {
                **candidate.metadata,
                "validation": {
                    "ok": False,
                    "hard_reject_reasons": validation.hard_reject_reasons,
                    "warnings": validation.warnings,
                    "stats": validation.stats,
                },
            }
            output.append(CorpusCandidate.from_dict(payload))
            continue

        asset_path = _asset_path(assets_dir, candidate, svg_string)
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_text(svg_string)
        payload = candidate.to_dict()
        payload["fetch_status"] = "fetched"
        payload["content_sha256"] = _sha256(svg_string)
        payload["local_path"] = str(asset_path)
        payload["metadata"] = {**candidate.metadata, "validation": {"ok": True, "warnings": validation.warnings, "stats": validation.stats}, "fetched_at": utc_now()}
        output.append(CorpusCandidate.from_dict(payload))
        stats["fetched"] += 1

    paths["root"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["candidates"], [candidate.to_dict() for candidate in output])
    manifest = build_manifest(corpus_id, str(paths["candidates"]), output, metadata={"stage": "indexed_fetch_validate", "stats": stats})
    paths["manifest"].write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    summary = paths["root"] / "fetch_validation_summary.json"
    summary.write_text(json.dumps({"corpus_id": corpus_id, "created_at": utc_now(), "stats": stats}, indent=2, sort_keys=True))
    return {"corpus_id": corpus_id, "candidates": str(paths["candidates"]), "manifest": str(paths["manifest"]), "summary": str(summary), "stats": stats}
