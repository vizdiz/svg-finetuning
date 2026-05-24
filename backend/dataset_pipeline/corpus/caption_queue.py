from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from backend.dataset_pipeline.corpus.schema import read_candidates, utc_now, write_jsonl


def build_model_caption_queue(
    *,
    input_candidates: Path,
    output_path: Path,
    min_priority: float = 0.75,
    min_caption_chars: int = 24,
    limit: int | None = None,
) -> dict[str, Any]:
    queue: list[dict[str, Any]] = []
    stats = {"scanned": 0, "queued": 0, "skipped_not_fetched": 0, "skipped_low_priority": 0, "skipped_has_caption": 0}

    for candidate in read_candidates(input_candidates):
        stats["scanned"] += 1
        if candidate.fetch_status not in {"fetched", "not_required"}:
            stats["skipped_not_fetched"] += 1
            continue
        if candidate.priority_score < min_priority:
            stats["skipped_low_priority"] += 1
            continue
        if len(candidate.caption.strip()) >= min_caption_chars:
            stats["skipped_has_caption"] += 1
            continue
        asset_uri = candidate.s3_uri or candidate.local_path
        if not asset_uri and candidate.source == "synthetic":
            asset_uri = candidate.uri
        if not asset_uri:
            stats["skipped_not_fetched"] += 1
            continue
        queue.append(
            {
                "candidate_id": candidate.candidate_id,
                "source": candidate.source,
                "asset_uri": asset_uri,
                "title": candidate.title,
                "license": candidate.license,
                "priority_score": candidate.priority_score,
                "metadata": candidate.metadata,
                "reason": "high_priority_missing_or_short_caption",
                "created_at": utc_now(),
            }
        )
        stats["queued"] += 1
        if limit is not None and stats["queued"] >= limit:
            break

    write_jsonl(output_path, queue)
    summary = {
        "created_at": utc_now(),
        "input_candidates": str(input_candidates),
        "output_path": str(output_path),
        "min_priority": min_priority,
        "min_caption_chars": min_caption_chars,
        "stats": stats,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
