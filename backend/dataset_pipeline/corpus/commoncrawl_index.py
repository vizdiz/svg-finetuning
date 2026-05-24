from __future__ import annotations

from pathlib import Path
import gzip
import io
import json
import zlib
from typing import Any, Iterable
from urllib.parse import quote

import httpx
from backend.dataset_pipeline.corpus.schema import utc_now


COMMONCRAWL_BUCKET = "commoncrawl"
COMMONCRAWL_INDEX_PREFIX = "cc-index/collections"


def _parse_cdxj_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text:
        return None
    if text.startswith("{"):
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    parts = text.split(" ", 2)
    if len(parts) < 3:
        return None
    try:
        payload = json.loads(parts[2])
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        payload.setdefault("urlkey", parts[0])
        payload.setdefault("timestamp", parts[1])
        return payload
    return None


def _svg_score(row: dict[str, Any]) -> float:
    score = 0.1
    url = str(row.get("url") or row.get("target_uri") or "")
    mime = str(row.get("mime") or row.get("content_mime_type") or "")
    detected = str(row.get("mime-detected") or row.get("content_mime_detected") or "")
    if url.lower().endswith(".svg") or ".svg?" in url.lower():
        score += 0.5
    if "svg" in mime.lower():
        score += 0.3
    if "svg" in detected.lower():
        score += 0.1
    if str(row.get("status") or row.get("fetch_status") or "") in {"200", 200}:
        score += 0.1
    return min(score, 1.0)


def _annotate_svg_row(
    row: dict[str, Any], crawl_id: str, index_file: str
) -> dict[str, Any] | None:
    url = str(row.get("url") or "")
    mime = str(row.get("mime") or row.get("content_mime_type") or "")
    if not (url.lower().endswith(".svg") or ".svg?" in url.lower() or "svg" in mime.lower()):
        return None
    filename = str(row.get("filename") or row.get("warc_filename") or "")
    if filename and not filename.startswith("s3://"):
        row["warc_uri"] = f"s3://{COMMONCRAWL_BUCKET}/{filename}"
    elif filename:
        row["warc_uri"] = filename
    row["score"] = _svg_score(row)
    row["crawl_id"] = crawl_id
    row["index_file"] = index_file
    return row


_CC_MAX_INDEX_PARTITIONS = 300


def _index_file_candidates(
    s3_client: Any | None, crawl_id: str, *, max_files: int | None = None
) -> list[str]:
    prefix = f"{COMMONCRAWL_INDEX_PREFIX}/{crawl_id}/indexes/"
    if s3_client is not None:
        response = s3_client.list_objects_v2(
            Bucket=COMMONCRAWL_BUCKET, Prefix=prefix, RequestPayer="requester"
        )
        keys = [item["Key"] for item in response.get("Contents", []) if item["Key"].endswith(".gz")]
        keys.sort()
    else:
        # Generate candidate key names; 404s are skipped in iter_commoncrawl_index_rows.
        n = max_files if max_files is not None else _CC_MAX_INDEX_PARTITIONS
        keys = [f"{prefix}cdx-{i:05d}.gz" for i in range(n)]
        return keys  # already ordered, max_files already applied
    if max_files is not None:
        return keys[:max_files]
    return keys


def _iter_lines_from_s3(s3_client: Any, key: str) -> Iterable[str]:
    resp = s3_client.get_object(Bucket=COMMONCRAWL_BUCKET, Key=key, RequestPayer="requester")
    with gzip.GzipFile(fileobj=io.BytesIO(resp["Body"].read())) as gz:
        for line_bytes in gz:
            yield line_bytes.decode("utf-8", errors="replace").rstrip()


def _iter_lines_from_https(key: str) -> Iterable[str]:
    """Stream-decompress a CDX gzip file from data.commoncrawl.org.

    Yields nothing on 404 (caller moves to next key automatically).
    Keeps peak memory to one 64KB decompressed chunk at a time.
    """
    url = f"https://data.commoncrawl.org/{quote(key)}"
    with httpx.stream("GET", url, timeout=120.0) as resp:
        if resp.status_code == 404:
            return  # empty generator — caller iterates over nothing and continues
        resp.raise_for_status()
        decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
        buf = b""
        for chunk in resp.iter_bytes(chunk_size=65536):
            buf += decompressor.decompress(chunk)
            # Split once per chunk — O(n) vs the naive while/index/slice O(n²).
            parts = buf.split(b"\n")
            buf = parts[-1]  # last element is the (possibly partial) current line
            for line_bytes in parts[:-1]:
                yield line_bytes.decode("utf-8", errors="replace").rstrip()
        buf += decompressor.flush()
        if buf.strip():
            yield buf.decode("utf-8", errors="replace").rstrip()


def iter_commoncrawl_index_rows(
    *,
    crawl_id: str,
    s3_client: Any | None = None,
    index_keys: list[str] | None = None,
    max_files: int | None = None,
    limit: int | None = None,
) -> Iterable[dict[str, Any]]:
    files = index_keys or _index_file_candidates(s3_client, crawl_id, max_files=max_files)
    yielded = 0
    for key in files:
        if s3_client is not None:
            lines: Iterable[str] = _iter_lines_from_s3(s3_client, key)
        else:
            lines = _iter_lines_from_https(key)

        for raw_line in lines:
            parsed = _parse_cdxj_line(raw_line)
            if parsed is None:
                continue
            row = _annotate_svg_row(parsed, crawl_id, key)
            if row is None:
                continue
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def fetch_commoncrawl_index_manifest(
    *,
    crawl_id: str,
    output_path: Path,
    s3_client: Any | None = None,
    index_keys: list[str] | None = None,
    max_files: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fout:
        for row in iter_commoncrawl_index_rows(
            crawl_id=crawl_id,
            s3_client=s3_client,
            index_keys=index_keys,
            max_files=max_files,
            limit=limit,
        ):
            fout.write(json.dumps(row) + "\n")
            count += 1
    summary = {
        "created_at": utc_now(),
        "crawl_id": crawl_id,
        "record_count": count,
        "max_files": max_files,
        "limit": limit,
        "output_path": str(output_path),
        "source": "commoncrawl",
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
