from __future__ import annotations

from pathlib import Path
import gzip
import io
import json
from typing import Any, Iterable

from backend.dataset_pipeline.corpus.schema import utc_now, write_jsonl


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


def _index_file_candidates(s3_client: Any, crawl_id: str, *, max_files: int | None = None) -> list[str]:
    prefix = f"{COMMONCRAWL_INDEX_PREFIX}/{crawl_id}/indexes/"
    response = s3_client.list_objects_v2(Bucket=COMMONCRAWL_BUCKET, Prefix=prefix)
    keys = [item["Key"] for item in response.get("Contents", []) if item["Key"].endswith(".gz")]
    keys.sort()
    if max_files is not None:
        return keys[:max_files]
    return keys


def iter_commoncrawl_index_rows(
    *,
    crawl_id: str,
    s3_client: Any,
    max_files: int | None = None,
    limit: int | None = None,
) -> Iterable[dict[str, Any]]:
    files = _index_file_candidates(s3_client, crawl_id, max_files=max_files)
    yielded = 0
    for key in files:
        response = s3_client.get_object(Bucket=COMMONCRAWL_BUCKET, Key=key)
        with gzip.GzipFile(fileobj=io.BytesIO(response["Body"].read())) as handle:
            for raw_line in handle.read().decode("utf-8", errors="replace").splitlines():
                row = _parse_cdxj_line(raw_line)
                if row is None:
                    continue
                url = str(row.get("url") or "")
                mime = str(row.get("mime") or row.get("content_mime_type") or "")
                if not (
                    url.lower().endswith(".svg")
                    or ".svg?" in url.lower()
                    or "svg" in mime.lower()
                ):
                    continue
                filename = str(row.get("filename") or row.get("warc_filename") or "")
                if filename and not filename.startswith("s3://"):
                    row["warc_uri"] = f"s3://{COMMONCRAWL_BUCKET}/{filename}"
                elif filename:
                    row["warc_uri"] = filename
                row["score"] = _svg_score(row)
                row["crawl_id"] = crawl_id
                row["index_file"] = key
                yield row
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def fetch_commoncrawl_index_manifest(
    *,
    crawl_id: str,
    output_path: Path,
    s3_client: Any | None = None,
    max_files: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    import boto3

    s3 = s3_client or boto3.client("s3")
    rows = list(iter_commoncrawl_index_rows(crawl_id=crawl_id, s3_client=s3, max_files=max_files, limit=limit))
    write_jsonl(output_path, rows)
    summary = {
        "created_at": utc_now(),
        "crawl_id": crawl_id,
        "record_count": len(rows),
        "max_files": max_files,
        "limit": limit,
        "output_path": str(output_path),
        "source": "commoncrawl",
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
