from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import re
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Iterable

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.corpus.schema import (
    CorpusCandidate,
    build_manifest,
    local_corpus_paths,
    stable_id,
    utc_now,
    write_jsonl,
)
from backend.dataset_pipeline.corpus.wikimedia_dump import (
    WikimediaDumpCandidate,
    candidate_from_image_row,
    iter_dump_candidates,
    iter_insert_rows,
    passes_config,
    score_dump_candidate,
    source_id as wikimedia_source_id,
    upload_url as wikimedia_upload_url,
)


ARXIV_ID_RE = re.compile(r"(?P<id>(?:\d{4}\.\d{4,5})(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)")


@dataclass(frozen=True, slots=True)
class BulkCandidateResult:
    corpus_id: str
    candidates: str
    manifest: str
    record_count: int
    source_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "candidates": self.candidates,
            "manifest": self.manifest,
            "record_count": self.record_count,
            "source_counts": self.source_counts,
        }


def _write_candidate_corpus(
    *,
    corpus_id: str,
    output_root: Path,
    candidates: list[CorpusCandidate],
    metadata: dict[str, Any],
) -> BulkCandidateResult:
    paths = local_corpus_paths(output_root, corpus_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["candidates"], [candidate.to_dict() for candidate in candidates])
    manifest = build_manifest(corpus_id, str(paths["candidates"]), candidates, metadata=metadata)
    paths["manifest"].write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return BulkCandidateResult(
        corpus_id=corpus_id,
        candidates=str(paths["candidates"]),
        manifest=str(paths["manifest"]),
        record_count=len(candidates),
        source_counts=manifest.source_counts,
    )


def rank_wikimedia_dump_candidates(
    *,
    dump_path: str,
    limit: int,
    config: ScrapeConfig | None = None,
) -> list[tuple[WikimediaDumpCandidate, float]]:
    if limit <= 0:
        return []
    scrape_config = config or ScrapeConfig(max_svgs_per_source=limit)
    heap: list[tuple[float, str, WikimediaDumpCandidate]] = []
    for candidate in iter_dump_candidates(dump_path, scrape_config):
        score = score_dump_candidate(candidate)
        item = (score, candidate.filename, candidate)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)
    return [(candidate, score) for score, _, candidate in sorted(heap, key=lambda item: (-item[0], item[1]))]


def _candidate_from_image_row(row: dict[int, str | None]) -> WikimediaDumpCandidate | None:
    return candidate_from_image_row(row)


def _passes_wikimedia_config(candidate: WikimediaDumpCandidate, config: ScrapeConfig) -> bool:
    return passes_config(candidate, config)


def _to_corpus_wikimedia_candidates(
    ranked: Iterable[tuple[WikimediaDumpCandidate, float]],
    *,
    dump_path: str,
    asset_base_url: str,
) -> list[CorpusCandidate]:
    candidates: list[CorpusCandidate] = []
    for candidate, score in ranked:
        source_url = wikimedia_upload_url(candidate.filename, asset_base_url)
        normalized_score = min(1.0, max(0.0, score / 140.0))
        candidates.append(
            CorpusCandidate(
                candidate_id=stable_id("wikimedia", candidate.filename, candidate.sha1, prefix="commons"),
                source="wikimedia",
                uri=source_url,
                target_format="diagram_ir",
                title=f"File:{candidate.filename}",
                license="commons-metadata-pending",
                provenance={
                    "bulk_source": Path(dump_path).name,
                    "discovery": "commonswiki image SQL dump",
                    "fetch_policy": "selected_original_only_after_ranking",
                    "source_id": wikimedia_source_id(candidate.filename),
                },
                metadata={
                    "filename": candidate.filename,
                    "size_bytes": candidate.size_bytes,
                    "width": candidate.width,
                    "height": candidate.height,
                    "timestamp": candidate.timestamp,
                    "sha1": candidate.sha1,
                    "rank_score": score,
                },
                priority_score=normalized_score,
                fetch_status="pending",
            )
        )
    return candidates


def make_line_aligned_shards(path: Path, shard_count: int) -> list[dict[str, Any]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    size = path.stat().st_size
    boundaries = [0]
    with path.open("rb") as handle:
        for shard_index in range(1, shard_count):
            handle.seek((size * shard_index) // shard_count)
            handle.readline()
            boundaries.append(handle.tell())
    boundaries.append(size)
    return [
        {
            "shard_index": index,
            "shard_count": shard_count,
            "path": str(path),
            "start": boundaries[index],
            "end": boundaries[index + 1],
            "size_bytes": boundaries[index + 1] - boundaries[index],
        }
        for index in range(shard_count)
    ]


def write_wikimedia_shard_plan(path: Path, dump_path: Path, shard_count: int) -> dict[str, Any]:
    plan = {
        "created_at": utc_now(),
        "dump_path": str(dump_path),
        "dump_size_bytes": dump_path.stat().st_size,
        "shards": make_line_aligned_shards(dump_path, shard_count),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def iter_wikimedia_dump_shard(path: Path, *, start: int, end: int) -> Iterable[WikimediaDumpCandidate]:
    with path.open("rb") as raw_handle:
        raw_handle.seek(start)
        while raw_handle.tell() < end:
            line = raw_handle.readline()
            if not line:
                break
            if not line.startswith(b"INSERT INTO `image` VALUES "):
                continue
            text = line.decode("utf-8", errors="replace")
            for row in iter_insert_rows(text):
                candidate = _candidate_from_image_row(row)
                if candidate is not None:
                    yield candidate


def rank_wikimedia_dump_shard(
    *,
    dump_path: Path,
    start: int,
    end: int,
    limit: int,
    config: ScrapeConfig | None = None,
) -> list[tuple[WikimediaDumpCandidate, float]]:
    if limit <= 0:
        return []
    scrape_config = config or ScrapeConfig(max_svgs_per_source=limit)
    heap: list[tuple[float, str, WikimediaDumpCandidate]] = []
    for candidate in iter_wikimedia_dump_shard(dump_path, start=start, end=end):
        if not _passes_wikimedia_config(candidate, scrape_config):
            continue
        score = score_dump_candidate(candidate)
        item = (score, candidate.filename, candidate)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)
    return [(candidate, score) for score, _, candidate in sorted(heap, key=lambda item: (-item[0], item[1]))]


def write_ranked_wikimedia_candidates(path: Path, ranked: list[tuple[WikimediaDumpCandidate, float]]) -> None:
    rows = []
    for candidate, score in ranked:
        rows.append(
            {
                "filename": candidate.filename,
                "size_bytes": candidate.size_bytes,
                "width": candidate.width,
                "height": candidate.height,
                "media_type": candidate.media_type,
                "major_mime": candidate.major_mime,
                "minor_mime": candidate.minor_mime,
                "timestamp": candidate.timestamp,
                "sha1": candidate.sha1,
                "rank_score": score,
            }
        )
    write_jsonl(path, rows)


def read_ranked_wikimedia_candidates(paths: Iterable[Path]) -> list[tuple[WikimediaDumpCandidate, float]]:
    ranked: list[tuple[WikimediaDumpCandidate, float]] = []
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ranked.append(
                (
                    WikimediaDumpCandidate(
                        filename=row["filename"],
                        size_bytes=int(row["size_bytes"]),
                        width=int(row["width"]),
                        height=int(row["height"]),
                        media_type=row.get("media_type", ""),
                        major_mime=row.get("major_mime", ""),
                        minor_mime=row.get("minor_mime", ""),
                        timestamp=row.get("timestamp", ""),
                        sha1=row.get("sha1", ""),
                    ),
                    float(row["rank_score"]),
                )
            )
    return ranked


def merge_ranked_wikimedia_candidates(
    ranked: Iterable[tuple[WikimediaDumpCandidate, float]],
    *,
    limit: int,
) -> list[tuple[WikimediaDumpCandidate, float]]:
    heap: list[tuple[float, str, WikimediaDumpCandidate]] = []
    seen: set[str] = set()
    for candidate, score in ranked:
        dedupe_key = candidate.sha1 or candidate.filename
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        item = (score, candidate.filename, candidate)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)
    return [(candidate, score) for score, _, candidate in sorted(heap, key=lambda item: (-item[0], item[1]))]


def wikimedia_candidates_from_dump(
    *,
    corpus_id: str,
    dump_path: str,
    output_root: Path,
    limit: int,
    config: ScrapeConfig | None = None,
) -> BulkCandidateResult:
    scrape_config = config or ScrapeConfig(max_svgs_per_source=limit)
    ranked = rank_wikimedia_dump_candidates(dump_path=dump_path, limit=limit, config=scrape_config)
    return wikimedia_candidates_from_ranked(
        corpus_id=corpus_id,
        ranked=ranked,
        output_root=output_root,
        dump_path=dump_path,
        asset_base_url=scrape_config.wikimedia_asset_base_url,
        metadata={
            "source": "wikimedia",
            "bulk_only": True,
            "dump_path": dump_path,
            "selection": "ranked_sql_dump_candidates",
            "created_at": utc_now(),
        },
    )


def wikimedia_candidates_from_ranked(
    *,
    corpus_id: str,
    ranked: list[tuple[WikimediaDumpCandidate, float]],
    output_root: Path,
    dump_path: str,
    asset_base_url: str,
    metadata: dict[str, Any] | None = None,
) -> BulkCandidateResult:
    candidates = _to_corpus_wikimedia_candidates(
        ranked,
        dump_path=dump_path,
        asset_base_url=asset_base_url,
    )
    return _write_candidate_corpus(
        corpus_id=corpus_id,
        output_root=output_root,
        candidates=candidates,
        metadata=metadata or {
            "source": "wikimedia",
            "bulk_only": True,
            "dump_path": dump_path,
            "selection": "ranked_sql_dump_candidates",
            "created_at": utc_now(),
        },
    )


def extract_arxiv_id(value: str) -> str:
    match = ARXIV_ID_RE.search(value)
    if not match:
        raise ValueError(f"could not find arXiv id in {value!r}")
    return match.group("id")


def arxiv_source_month(arxiv_id: str) -> str:
    clean = arxiv_id.split("v", 1)[0]
    if "/" in clean:
        return "legacy"
    return clean[:4]


def arxiv_bulk_candidates(
    *,
    corpus_id: str,
    arxiv_ids: Iterable[str],
    output_root: Path,
    target_format: str = "diagram_ir",
) -> BulkCandidateResult:
    candidates: list[CorpusCandidate] = []
    seen: set[str] = set()
    for raw_id in arxiv_ids:
        arxiv_id = extract_arxiv_id(raw_id.strip())
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)
        source_month = arxiv_source_month(arxiv_id)
        candidates.append(
            CorpusCandidate(
                candidate_id=stable_id("arxiv", arxiv_id, prefix="arxiv"),
                source="arxiv",
                uri=f"arxiv://source/{arxiv_id}",
                target_format=target_format,
                title=f"arXiv:{arxiv_id}",
                license="arxiv-license-metadata-pending",
                provenance={
                    "bulk_source": "arxiv requester-pays source tarballs",
                    "source_month": source_month,
                    "fetch_policy": "s3_source_tarball_only",
                },
                metadata={
                    "arxiv_id": arxiv_id,
                    "source_month": source_month,
                    "expected_artifacts": ["tex", "tikz", "svg", "pdf_figures"],
                },
                priority_score=0.8,
                fetch_status="pending",
            )
        )
    return _write_candidate_corpus(
        corpus_id=corpus_id,
        output_root=output_root,
        candidates=candidates,
        metadata={
            "source": "arxiv",
            "bulk_only": True,
            "selection": "arxiv_id_seed_list",
            "created_at": utc_now(),
        },
    )


def arxiv_ids_from_local_paths(paths: Iterable[Path]) -> list[str]:
    ids: list[str] = []
    for path in paths:
        try:
            ids.append(extract_arxiv_id(path.name))
        except ValueError:
            continue
    return ids


def _priority_from_row(row: dict[str, Any], default: float = 0.5) -> float:
    for key in ("priority_score", "score", "rank_score"):
        if key in row:
            return min(1.0, max(0.0, float(row[key])))
    return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def github_candidates_from_index(
    *,
    corpus_id: str,
    index_path: Path,
    output_root: Path,
    limit: int,
) -> BulkCandidateResult:
    candidates: list[CorpusCandidate] = []
    seen: set[str] = set()
    rows = sorted(_read_jsonl(index_path), key=lambda row: _priority_from_row(row), reverse=True)
    for row in rows:
        if len(candidates) >= limit:
            break
        uri = str(row.get("raw_url") or row.get("uri") or row.get("blob_url") or "")
        if not uri:
            continue
        repo = str(row.get("repo") or row.get("repository") or "")
        path = str(row.get("path") or row.get("filepath") or "")
        dedupe_key = row.get("sha") or row.get("blob_sha") or uri
        if dedupe_key in seen:
            continue
        seen.add(str(dedupe_key))
        candidates.append(
            CorpusCandidate(
                candidate_id=stable_id("github", dedupe_key, uri, prefix="github"),
                source="github",
                uri=uri,
                target_format=str(row.get("target_format") or "raw_svg"),
                title=str(row.get("title") or path or uri.rsplit("/", 1)[-1]),
                caption=str(row.get("caption") or ""),
                license=str(row.get("license") or "github-license-metadata-pending"),
                provenance={
                    "bulk_source": str(index_path),
                    "discovery": "github_public_index",
                    "fetch_policy": "selected_raw_blob_only_after_ranking",
                    "repo": repo,
                    "path": path,
                },
                metadata={k: v for k, v in row.items() if k not in {"raw_url", "uri", "blob_url"}},
                priority_score=_priority_from_row(row, default=0.55),
                fetch_status="pending",
            )
        )
    return _write_candidate_corpus(
        corpus_id=corpus_id,
        output_root=output_root,
        candidates=candidates,
        metadata={
            "source": "github",
            "bulk_only": True,
            "selection": "public_index_ranked_svg_candidates",
            "index_path": str(index_path),
            "created_at": utc_now(),
        },
    )


def commoncrawl_candidates_from_index(
    *,
    corpus_id: str,
    index_path: Path,
    output_root: Path,
    limit: int,
) -> BulkCandidateResult:
    candidates: list[CorpusCandidate] = []
    seen: set[str] = set()
    rows = sorted(_read_jsonl(index_path), key=lambda row: _priority_from_row(row), reverse=True)
    for row in rows:
        if len(candidates) >= limit:
            break
        uri = str(row.get("warc_uri") or row.get("uri") or "")
        url = str(row.get("url") or row.get("target_uri") or "")
        if not uri and not url:
            continue
        dedupe_key = row.get("digest") or row.get("sha256") or uri or url
        if dedupe_key in seen:
            continue
        seen.add(str(dedupe_key))
        parsed = urlparse(url)
        candidates.append(
            CorpusCandidate(
                candidate_id=stable_id("commoncrawl", dedupe_key, uri, url, prefix="cc"),
                source="commoncrawl",
                uri=uri or url,
                target_format=str(row.get("target_format") or "raw_svg"),
                title=str(row.get("title") or parsed.path.rsplit("/", 1)[-1] or url),
                caption=str(row.get("caption") or ""),
                license=str(row.get("license") or "web-license-metadata-pending"),
                provenance={
                    "bulk_source": str(index_path),
                    "discovery": "common_crawl_index_wat_warc",
                    "fetch_policy": "read_warc_record_no_origin_request",
                    "source_url": url,
                    "host": parsed.netloc,
                },
                metadata={k: v for k, v in row.items() if k not in {"warc_uri", "uri"}},
                priority_score=_priority_from_row(row, default=0.45),
                fetch_status="pending",
            )
        )
    return _write_candidate_corpus(
        corpus_id=corpus_id,
        output_root=output_root,
        candidates=candidates,
        metadata={
            "source": "commoncrawl",
            "bulk_only": True,
            "selection": "index_ranked_svg_records",
            "index_path": str(index_path),
            "created_at": utc_now(),
        },
    )
