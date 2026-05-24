from __future__ import annotations

from pathlib import Path
import gzip
import json
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from backend.dataset_pipeline.corpus.schema import CorpusCandidate, read_candidates, utc_now


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got {uri}")
    bucket, key = uri[len("s3://") :].split("/", 1)
    return bucket, key


def build_s3_client(*, max_attempts: int = 10) -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        config=Config(
            retries={"max_attempts": max_attempts, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=120,
        ),
    )


def _source_archive_suffix(key: str) -> str:
    lower_key = key.lower()
    if lower_key.endswith(".tar.gz"):
        return ".tar.gz"
    if lower_key.endswith(".tgz"):
        return ".tgz"
    if lower_key.endswith(".tar"):
        return ".tar"
    return Path(key).suffix or ".tar"


def _arxiv_https_download(arxiv_id: str, dest: Path, *, delay_s: float = 3.0) -> None:
    url = f"https://arxiv.org/src/{arxiv_id}"
    time.sleep(delay_s)
    resp = httpx.get(url, follow_redirects=True, timeout=120.0)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def download_arxiv_source_batch(
    *,
    input_candidates: Path,
    destination_root: Path,
    s3_client: Any | None = None,
    limit: int | None = None,
    request_payer: bool = True,
    https_fallback: bool = True,
    https_delay_s: float = 3.0,
) -> dict[str, Any]:
    destination_root.mkdir(parents=True, exist_ok=True)
    stats = {
        "total": 0, "downloaded": 0, "skipped_existing": 0,
        "missing_s3_uri": 0, "https_fallback": 0, "failed": 0, "limited": False,
    }
    failures: list[dict[str, str]] = []
    _s3: Any | None = None

    def _get_s3() -> Any:
        nonlocal _s3
        if _s3 is None:
            _s3 = s3_client or build_s3_client()
        return _s3

    for candidate in read_candidates(input_candidates):
        if candidate.source != "arxiv":
            continue
        if limit is not None and stats["total"] >= limit:
            stats["limited"] = True
            break
        stats["total"] += 1
        arxiv_id = str(candidate.metadata.get("arxiv_id") or "")
        metadata_uri = str(candidate.metadata.get("source_s3_uri") or "")
        candidate_uri = str(candidate.uri)
        s3_uri = metadata_uri or (candidate_uri if candidate_uri.startswith("s3://") else "")

        if s3_uri:
            try:
                bucket, key = parse_s3_uri(s3_uri)
                filename = arxiv_id or Path(key).name
                suffix = _source_archive_suffix(key)
                basename = str(filename).replace("/", "_")
                dest = destination_root / (basename if basename.endswith(suffix) else f"{basename}{suffix}")
                if dest.exists() and dest.stat().st_size > 0:
                    stats["skipped_existing"] += 1
                    continue
                extra_args = {"RequestPayer": "requester"} if request_payer else None
                if extra_args:
                    _get_s3().download_file(bucket, key, str(dest), ExtraArgs=extra_args)
                else:
                    _get_s3().download_file(bucket, key, str(dest))
                stats["downloaded"] += 1
                continue
            except Exception as exc:
                if not https_fallback or not arxiv_id:
                    stats["failed"] += 1
                    failures.append({"candidate_id": candidate.candidate_id, "reason": f"{type(exc).__name__}: {exc}"})
                    continue
                # fall through to HTTPS

        if https_fallback and arxiv_id:
            dest = destination_root / f"{arxiv_id.replace('/', '_')}.tar.gz"
            if dest.exists() and dest.stat().st_size > 0:
                stats["skipped_existing"] += 1
                continue
            try:
                _arxiv_https_download(arxiv_id, dest, delay_s=https_delay_s)
                stats["downloaded"] += 1
                stats["https_fallback"] += 1
            except Exception as exc:
                stats["failed"] += 1
                failures.append({"candidate_id": candidate.candidate_id, "reason": f"https: {type(exc).__name__}: {exc}"})
        else:
            stats["missing_s3_uri"] += 1

    summary = {
        "created_at": utc_now(),
        "input_candidates": str(input_candidates),
        "destination_root": str(destination_root),
        "stats": stats,
        "failures": failures[:100],
    }
    (destination_root / "download_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _read_s3_range(s3_client: Any, uri: str, offset: int, length: int) -> bytes:
    bucket, key = parse_s3_uri(uri)
    response = s3_client.get_object(Bucket=bucket, Key=key, Range=f"bytes={offset}-{offset + length - 1}")
    return response["Body"].read()


def commoncrawl_http_to_s3_uri(uri: str) -> str:
    if uri.startswith("s3://"):
        return uri
    parsed = urlparse(uri)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc.endswith("commoncrawl.org"):
        raise ValueError("Common Crawl record must reference a commoncrawl.org HTTP URI or s3:// URI")
    path = parsed.path.lstrip("/")
    if parsed.netloc == "data.commoncrawl.org":
        return f"s3://commoncrawl/{path}"
    return f"s3://{path}"


def _extract_svg_from_warc_payload(payload: bytes) -> str:
    try:
        raw = gzip.decompress(payload)
    except OSError:
        raw = payload
    text = raw.decode("utf-8", errors="replace")
    start = text.lower().find("<svg")
    end = text.lower().rfind("</svg>")
    if start < 0 or end < 0:
        raise ValueError("svg element not found in WARC payload")
    return text[start : end + len("</svg>")]


def read_commoncrawl_warc_record(candidate: CorpusCandidate, *, s3_client: Any | None = None) -> str:
    s3 = s3_client or build_s3_client()
    warc_uri = str(candidate.metadata.get("warc_uri") or candidate.uri)
    offset = int(candidate.metadata.get("offset") or candidate.metadata.get("warc_offset") or 0)
    length = int(candidate.metadata.get("length") or candidate.metadata.get("warc_length") or 0)
    if not warc_uri.startswith("s3://"):
        warc_uri = commoncrawl_http_to_s3_uri(warc_uri)
    if length <= 0:
        bucket, key = parse_s3_uri(warc_uri)
        response = s3.get_object(Bucket=bucket, Key=key)
        payload = response["Body"].read()
    else:
        payload = _read_s3_range(s3, warc_uri, offset, length)
    return _extract_svg_from_warc_payload(payload)
