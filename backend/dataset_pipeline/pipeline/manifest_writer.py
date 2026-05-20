"""
dataset_pipeline.pipeline.manifest_writer

Writes split JSONL datasets and the final manifest for a completed pipeline run.

The S3 manifest write is intentionally last because uploading
train/dataset_manifest.json is the retraining trigger.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import boto3

from backend.dataset_pipeline.config import PipelineConfig
from backend.dataset_pipeline.processing.captioner import TrainingRecord

logger = logging.getLogger(__name__)

_TRAIN_KEY = "train/dataset.jsonl"
_VAL_KEY = "val/dataset.jsonl"
_MANIFEST_KEY = "train/dataset_manifest.json"


def _assign_split(record: TrainingRecord) -> str:
    h = int(hashlib.md5(record.id.encode()).hexdigest(), 16) % 100
    split = "val" if h < 10 else "train"
    record.split = split
    return split


def _split_records(records: list[TrainingRecord]) -> tuple[list[TrainingRecord], list[TrainingRecord]]:
    train_records: list[TrainingRecord] = []
    val_records: list[TrainingRecord] = []

    for record in records:
        if _assign_split(record) == "val":
            val_records.append(record)
        else:
            train_records.append(record)

    return train_records, val_records


def _jsonl(records: list[TrainingRecord]) -> str:
    return "".join(json.dumps(asdict(record)) + "\n" for record in records)


def _source_counts(records: list[TrainingRecord]) -> dict[str, int]:
    return dict(Counter(record.source for record in records))


def _build_manifest(
    records: list[TrainingRecord],
    train_count: int,
    val_count: int,
    train_uri: str,
    val_uri: str,
) -> dict:
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return {
        "version": version,
        "dataset_id": version,
        "record_count": len(records),
        "files": [train_uri, val_uri],
        "num_train": train_count,
        "num_val": val_count,
        "total": len(records),
        "sources": _source_counts(records),
        "train_s3_uri": train_uri,
        "val_s3_uri": val_uri,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


def _summary_rows(manifest: dict) -> list[tuple[str, int]]:
    rows = [
        ("Total", manifest["total"]),
        ("Train", manifest["num_train"]),
        ("Val", manifest["num_val"]),
    ]

    sources = manifest["sources"]
    ordered_sources = ["arxiv", "wikipedia"]
    ordered_sources.extend(source for source in sorted(sources) if source not in ordered_sources)
    rows.extend((source, sources.get(source, 0)) for source in ordered_sources)
    return rows


def _log_summary_table(manifest: dict) -> None:
    rows = _summary_rows(manifest)
    metric_width = max(len("Metric"), *(len(metric) for metric, _ in rows))
    count_width = max(len("Count"), *(len(str(count)) for _, count in rows))

    top = f"┌{'─' * (metric_width + 2)}┬{'─' * (count_width + 2)}┐"
    header = f"│ {'Metric'.ljust(metric_width)} │ {'Count'.ljust(count_width)} │"
    sep = f"├{'─' * (metric_width + 2)}┼{'─' * (count_width + 2)}┤"
    bottom = f"└{'─' * (metric_width + 2)}┴{'─' * (count_width + 2)}┘"
    body = [
        f"│ {metric.ljust(metric_width)} │ {str(count).ljust(count_width)} │"
        for metric, count in rows
    ]

    logger.info("\n%s\n%s\n%s\n%s\n%s", top, header, sep, "\n".join(body), bottom)


def write_manifest(records: list[TrainingRecord], config: PipelineConfig) -> str:
    train_records, val_records = _split_records(records)

    train_s3_uri = f"s3://{config.s3_data_bucket}/{_TRAIN_KEY}"
    val_s3_uri = f"s3://{config.s3_data_bucket}/{_VAL_KEY}"
    manifest_s3_uri = f"s3://{config.s3_data_bucket}/{_MANIFEST_KEY}"

    manifest = _build_manifest(
        records=records,
        train_count=len(train_records),
        val_count=len(val_records),
        train_uri=train_s3_uri,
        val_uri=val_s3_uri,
    )

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=config.s3_data_bucket,
        Key=_TRAIN_KEY,
        Body=_jsonl(train_records).encode(),
        ContentType="application/jsonl",
    )
    s3.put_object(
        Bucket=config.s3_data_bucket,
        Key=_VAL_KEY,
        Body=_jsonl(val_records).encode(),
        ContentType="application/jsonl",
    )

    _log_summary_table(manifest)

    # TRIGGER: this write fires the retraining Lambda
    s3.put_object(
        Bucket=config.s3_data_bucket,
        Key=_MANIFEST_KEY,
        Body=json.dumps(manifest, indent=2).encode(),
        ContentType="application/json",
    )

    return manifest_s3_uri


def dry_run_write(records: list[TrainingRecord], output_dir: str) -> str:
    train_records, val_records = _split_records(records)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train.jsonl"
    val_path = out / "val.jsonl"
    manifest_path = out / "manifest.json"

    train_path.write_text(_jsonl(train_records))
    val_path.write_text(_jsonl(val_records))

    manifest = _build_manifest(
        records=records,
        train_count=len(train_records),
        val_count=len(val_records),
        train_uri=str(train_path),
        val_uri=str(val_path),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))

    _log_summary_table(manifest)

    return str(manifest_path)
