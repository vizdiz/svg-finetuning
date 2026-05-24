from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tarfile
import tempfile
from typing import Any

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.corpus.schema import CorpusCandidate, build_manifest, local_corpus_paths, read_candidates, stable_id, utc_now, write_jsonl
from backend.dataset_pipeline.processing.validator import validate_svg_detailed


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in name).strip("_") or "asset.svg"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _candidate_source_path(candidate: CorpusCandidate, source_root: Path) -> Path:
    if "source_path" in candidate.metadata:
        return Path(candidate.metadata["source_path"])
    arxiv_id = str(candidate.metadata.get("arxiv_id") or candidate.title.replace("arXiv:", ""))
    candidates = [
        source_root / f"{arxiv_id}.tar",
        source_root / f"{arxiv_id}.tar.gz",
        source_root / f"{arxiv_id.replace('/', '_')}.tar",
        source_root / f"{arxiv_id.replace('/', '_')}.tar.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no source tarball for {arxiv_id} under {source_root}")


def _iter_svg_members(tar_path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".svg"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            records.append((member.name, handle.read().decode("utf-8", errors="replace")))
    return records


def extract_arxiv_sources(
    *,
    corpus_id: str,
    input_candidates: Path,
    source_root: Path,
    output_root: Path,
    config: ScrapeConfig | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    scrape_config = config or ScrapeConfig()
    source_candidates = read_candidates(input_candidates)
    paths = local_corpus_paths(output_root, corpus_id)
    assets_dir = paths["assets"] / "arxiv"
    assets_dir.mkdir(parents=True, exist_ok=True)

    output: list[CorpusCandidate] = []
    stats = {"papers": len(source_candidates), "assets": 0, "rejected": 0, "failed": 0, "limited": False}
    processed = 0

    for candidate in source_candidates:
        if candidate.source != "arxiv":
            continue
        if limit is not None and processed >= limit:
            stats["limited"] = True
            break
        processed += 1
        try:
            tar_path = _candidate_source_path(candidate, source_root)
            svgs = _iter_svg_members(tar_path)
        except Exception as exc:
            stats["failed"] += 1
            payload = candidate.to_dict()
            payload["fetch_status"] = "failed"
            payload["metadata"] = {**candidate.metadata, "extract_error": str(exc), "extract_error_type": type(exc).__name__}
            output.append(CorpusCandidate.from_dict(payload))
            continue

        for member_name, svg_string in svgs:
            validation = validate_svg_detailed(svg_string, scrape_config)
            asset_id = stable_id(candidate.candidate_id, member_name, prefix="arxiv-svg")
            if not validation.ok:
                stats["rejected"] += 1
                output.append(
                    CorpusCandidate(
                        candidate_id=asset_id,
                        source="arxiv",
                        uri=f"{candidate.uri}#{member_name}",
                        target_format="raw_svg",
                        title=f"{candidate.title}:{member_name}",
                        license=candidate.license,
                        provenance={**candidate.provenance, "source_tarball": str(tar_path), "member": member_name},
                        metadata={
                            **candidate.metadata,
                            "validation": {
                                "ok": False,
                                "hard_reject_reasons": validation.hard_reject_reasons,
                                "warnings": validation.warnings,
                                "stats": validation.stats,
                            },
                        },
                        priority_score=candidate.priority_score,
                        fetch_status="rejected",
                    )
                )
                continue
            asset_path = assets_dir / f"{asset_id}-{_safe_name(Path(member_name).name)}"
            asset_path.write_text(svg_string)
            stats["assets"] += 1
            output.append(
                CorpusCandidate(
                    candidate_id=asset_id,
                    source="arxiv",
                    uri=f"{candidate.uri}#{member_name}",
                    target_format="raw_svg",
                    title=f"{candidate.title}:{member_name}",
                    caption=candidate.caption,
                    license=candidate.license,
                    provenance={**candidate.provenance, "source_tarball": str(tar_path), "member": member_name},
                    metadata={**candidate.metadata, "validation": {"ok": True, "warnings": validation.warnings, "stats": validation.stats}},
                    priority_score=candidate.priority_score,
                    fetch_status="fetched",
                    content_sha256=_sha256(svg_string),
                    local_path=str(asset_path),
                )
            )

    paths["root"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["candidates"], [candidate.to_dict() for candidate in output])
    manifest = build_manifest(corpus_id, str(paths["candidates"]), output, metadata={"source": "arxiv", "stage": "source_tarball_extract", "stats": stats})
    paths["manifest"].write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    summary = paths["root"] / "extract_summary.json"
    summary.write_text(json.dumps({"corpus_id": corpus_id, "created_at": utc_now(), "stats": stats}, indent=2, sort_keys=True))
    return {"corpus_id": corpus_id, "candidates": str(paths["candidates"]), "manifest": str(paths["manifest"]), "summary": str(summary), "stats": stats}


def download_s3_source_tarball(s3_uri: str, destination: Path) -> Path:
    import boto3

    if not s3_uri.startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got {s3_uri}")
    bucket, key = s3_uri[len("s3://") :].split("/", 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(destination))
    return destination


def extract_one_s3_tarball_to_temp(s3_uri: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="arxiv-source-"))
    return download_s3_source_tarball(s3_uri, tmp / Path(s3_uri).name)
