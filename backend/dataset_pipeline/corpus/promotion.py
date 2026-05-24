from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from backend.dataset_pipeline.corpus.s3_orchestration import parse_s3_uri
from backend.dataset_pipeline.corpus.s3_orchestration import build_s3_client
from backend.training.dataset_interface import DatasetManifest, MANIFEST_KEY, write_manifest_to_s3


def _load_eval_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalise_s3_prefix(prefix: str) -> str:
    if not prefix.startswith("s3://"):
        raise ValueError("s3_dataset_prefix must be an s3:// URI")
    return prefix.rstrip("/")


def promote_dataset_manifest(
    *,
    product_manifest_path: Path,
    eval_report_path: Path,
    data_bucket: str,
    s3_dataset_prefix: str,
    s3_client: Any | None = None,
    approved: bool = False,
) -> dict[str, Any]:
    if not approved:
        raise ValueError("promotion requires approved=True")
    eval_report = _load_eval_report(eval_report_path)
    if not bool(eval_report.get("passed")):
        raise ValueError("eval report did not pass; refusing to promote")

    s3 = s3_client or build_s3_client()
    raw_manifest = json.loads(product_manifest_path.read_text())
    dataset_prefix = _normalise_s3_prefix(s3_dataset_prefix)
    prefix_bucket, prefix_key = parse_s3_uri(dataset_prefix)
    uploaded_files: list[str] = []

    files: list[str] = []
    for file_ref in raw_manifest.get("files", []):
        file_ref = str(file_ref)
        if file_ref.startswith("s3://"):
            files.append(file_ref)
            continue
        local_path = Path(file_ref)
        if not local_path.exists():
            raise FileNotFoundError(local_path)
        key = f"{prefix_key}/{local_path.name}" if prefix_key else local_path.name
        s3.put_object(
            Bucket=prefix_bucket,
            Key=key,
            Body=local_path.read_bytes(),
            ContentType="application/jsonl",
        )
        s3_uri = f"s3://{prefix_bucket}/{key}"
        files.append(s3_uri)
        uploaded_files.append(s3_uri)

    manifest = DatasetManifest(
        dataset_id=str(raw_manifest["dataset_id"]),
        created_at=str(raw_manifest["created_at"]),
        record_count=int(raw_manifest["record_count"]),
        files=files,
        schema_version=str(raw_manifest.get("schema_version", "1.0")),
        split=str(raw_manifest.get("split", "train")),
        metadata={
            **dict(raw_manifest.get("metadata", {}) or {}),
            "promoted_from": str(product_manifest_path),
            "eval_report": str(eval_report_path),
            "eval_summary": {
                "schema_validity_rate": eval_report.get("schema_validity_rate"),
                "compilability_rate": eval_report.get("compilability_rate"),
                "render_validity_rate": eval_report.get("render_validity_rate"),
            },
        },
    )
    write_manifest_to_s3(manifest, data_bucket, s3_client=s3)
    return {
        "dataset_id": manifest.dataset_id,
        "record_count": manifest.record_count,
        "files": files,
        "uploaded_files": uploaded_files,
        "promoted_manifest_uri": f"s3://{data_bucket}/{MANIFEST_KEY}",
    }
