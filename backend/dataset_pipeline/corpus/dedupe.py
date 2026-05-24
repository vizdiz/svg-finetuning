from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from typing import Any

from backend.dataset_pipeline.corpus.schema import CorpusCandidate, build_manifest, local_corpus_paths, read_candidates, utc_now, write_jsonl


def dedupe_candidates(
    *,
    corpus_id: str,
    input_candidates: list[Path],
    output_root: Path,
) -> dict[str, Any]:
    seen: set[str] = set()
    deduped: list[CorpusCandidate] = []
    stats = Counter()

    for path in input_candidates:
        for candidate in read_candidates(path):
            key = candidate.content_sha256 or candidate.metadata.get("normalized_svg_hash") or candidate.uri
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(str(key))
            deduped.append(candidate)
            stats["kept"] += 1

    paths = local_corpus_paths(output_root, corpus_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["candidates"], [candidate.to_dict() for candidate in deduped])
    manifest = build_manifest(corpus_id, str(paths["candidates"]), deduped, metadata={"stage": "dedupe", "stats": dict(stats), "inputs": [str(path) for path in input_candidates]})
    paths["manifest"].write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    summary = paths["root"] / "dedupe_summary.json"
    summary.write_text(json.dumps({"corpus_id": corpus_id, "created_at": utc_now(), "stats": dict(stats)}, indent=2, sort_keys=True))
    return {"corpus_id": corpus_id, "candidates": str(paths["candidates"]), "manifest": str(paths["manifest"]), "summary": str(summary), "stats": dict(stats)}
