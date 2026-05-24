from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

from backend.dataset_pipeline.corpus.schema import CorpusCandidate, local_corpus_paths, read_candidates, utc_now, write_jsonl
from backend.dataset_pipeline.processing.ir_labeler import label_raw_svg_ir
from backend.dataset_pipeline.processing.normalizer import normalize_svg
from backend.dataset_pipeline.processing.types import RawSVG


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _prompt_for(candidate: CorpusCandidate) -> str:
    if candidate.caption:
        return candidate.caption
    title = candidate.title or candidate.metadata.get("filename") or candidate.candidate_id
    return f"Draw a technical SVG diagram matching {title}."


def normalize_fetched_assets(
    *,
    corpus_id: str,
    input_candidates: Path,
    output_root: Path,
    require_ir: bool = False,
) -> dict[str, Any]:
    paths = local_corpus_paths(output_root, corpus_id)
    records: list[dict[str, Any]] = []
    stats = {"seen": 0, "written": 0, "missing_asset": 0, "ir_failed": 0}

    for candidate in read_candidates(input_candidates):
        if candidate.fetch_status not in {"fetched", "not_required"}:
            continue
        stats["seen"] += 1
        svg = ""
        if candidate.local_path:
            path = Path(candidate.local_path)
            if not path.exists():
                stats["missing_asset"] += 1
                continue
            svg = path.read_text()
            try:
                svg = normalize_svg(svg)
            except Exception:
                pass
        diagram_ir = candidate.metadata.get("diagram_ir")
        if svg and diagram_ir is None:
            raw = RawSVG(svg_string=svg, source_url=candidate.uri, source_id=candidate.candidate_id, domain=candidate.source, metadata=dict(candidate.metadata))
            result = label_raw_svg_ir(raw)
            if result.accepted:
                diagram_ir = raw.metadata.get("diagram_ir")
            else:
                stats["ir_failed"] += 1
        if require_ir and diagram_ir is None:
            continue
        records.append(
            {
                "id": candidate.candidate_id,
                "prompt": _prompt_for(candidate),
                "svg": svg,
                "diagram_ir": diagram_ir,
                "source": candidate.source,
                "split": "",
                "metadata": {
                    **candidate.metadata,
                    "target_format": "diagram_ir" if diagram_ir is not None else "raw_svg",
                    "content_sha256": candidate.content_sha256 or (_hash(svg) if svg else ""),
                    "candidate_uri": candidate.uri,
                    "license": candidate.license,
                    "provenance": candidate.provenance,
                },
            }
        )
        stats["written"] += 1

    paths["training"].mkdir(parents=True, exist_ok=True)
    output = paths["training"] / "records.jsonl"
    write_jsonl(output, records)
    summary = paths["root"] / "normalize_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"corpus_id": corpus_id, "created_at": utc_now(), "stats": stats, "records": str(output)}, indent=2, sort_keys=True))
    return {"corpus_id": corpus_id, "records": str(output), "summary": str(summary), "stats": stats}
