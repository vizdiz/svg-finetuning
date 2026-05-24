from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import hashlib

import boto3

from backend.dataset_pipeline.processing.captioner import TrainingRecord
from backend.training.canonical_ir import canonicalize_diagram_ir, compact_ir_json


BUCKET = "svg-finetuning-data-446224796301"


def _read_records(paths: list[Path]) -> list[TrainingRecord]:
    records: list[TrainingRecord] = []
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(TrainingRecord(**payload))
    return records


def _jsonl(records: list[TrainingRecord]) -> str:
    return "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records)


def _manifest(records: list[TrainingRecord], train_uri: str, val_uri: str, max_nodes: int, max_edges: int) -> dict:
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    dataset_id = f"canonical_arxiv_{len(records)}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    return {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "created_at": created_at,
        "record_count": len(records),
        "files": [train_uri, val_uri],
        "split": "train",
        "metadata": {
            "source": "captioned_arxiv_canonical_ir",
            "canonical_ir": True,
            "canonical_max_nodes": max_nodes,
            "canonical_max_edges": max_edges,
            "target_format": "diagram_ir",
        },
        "num_train": sum(1 for record in records if record.split == "train"),
        "num_val": sum(1 for record in records if record.split == "val"),
        "sources": {"arxiv": len(records)},
        "train_s3_uri": train_uri,
        "val_s3_uri": val_uri,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="pipeline_output/captioned_arxiv_199")
    parser.add_argument("--output-dir", default="pipeline_output/canonical_arxiv_197")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--s3-prefix", default="dry-run/canonical_arxiv_197")
    parser.add_argument("--max-nodes", type=int, default=24)
    parser.add_argument("--max-edges", type=int, default=48)
    parser.add_argument("--shortest", type=int, default=0)
    parser.add_argument("--profile", default="svg-finetuning")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_records([input_dir / "train.jsonl", input_dir / "val.jsonl"])
    reports = []
    canonical_records: list[TrainingRecord] = []
    for record in records:
        canonical, report = canonicalize_diagram_ir(
            record.diagram_ir,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
        )
        metadata = dict(record.metadata or {})
        metadata.pop("diagram_ir", None)
        metadata["canonical_ir_report"] = report.to_dict()
        metadata["canonical_ir_target_chars"] = len(compact_ir_json(canonical))
        canonical_records.append(
            TrainingRecord(
                id=record.id,
                prompt=record.prompt,
                completion=record.completion,
                source=record.source,
                split=record.split,
                metadata=metadata,
                diagram_ir=canonical.to_dict(),
            )
        )
        reports.append(report.to_dict())

    if args.shortest:
        canonical_records = sorted(
            canonical_records,
            key=lambda record: (
                len(record.prompt) + len(compact_ir_json(record.diagram_ir or {})),
                record.id,
            ),
        )[: args.shortest]
        for record in canonical_records:
            h = int(hashlib.md5(record.id.encode()).hexdigest(), 16) % 100
            record.split = "val" if h < 10 else "train"
        reports = [
            record.metadata["canonical_ir_report"]
            for record in canonical_records
            if "canonical_ir_report" in record.metadata
        ]

    train_records = [record for record in canonical_records if record.split == "train"]
    val_records = [record for record in canonical_records if record.split == "val"]
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    manifest_path = output_dir / "manifest.json"
    reports_path = output_dir / "canonical_reports.jsonl"

    train_path.write_text(_jsonl(train_records))
    val_path.write_text(_jsonl(val_records))
    reports_path.write_text("".join(json.dumps(report, sort_keys=True) + "\n" for report in reports))

    s3 = boto3.Session(profile_name=args.profile).client("s3")
    prefix = args.s3_prefix.strip("/")
    train_key = f"{prefix}/train.jsonl"
    val_key = f"{prefix}/val.jsonl"
    manifest_key = f"{prefix}/manifest.json"
    train_uri = f"s3://{args.bucket}/{train_key}"
    val_uri = f"s3://{args.bucket}/{val_key}"
    manifest = _manifest(canonical_records, train_uri, val_uri, args.max_nodes, args.max_edges)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    s3.put_object(Bucket=args.bucket, Key=train_key, Body=train_path.read_bytes(), ContentType="application/jsonl")
    s3.put_object(Bucket=args.bucket, Key=val_key, Body=val_path.read_bytes(), ContentType="application/jsonl")
    s3.put_object(Bucket=args.bucket, Key=manifest_key, Body=manifest_path.read_bytes(), ContentType="application/json")

    target_chars = [report["target_chars"] for report in reports]
    print(
        json.dumps(
            {
                "records": len(canonical_records),
                "train": len(train_records),
                "val": len(val_records),
                "manifest": f"s3://{args.bucket}/{manifest_key}",
                "target_chars_max": max(target_chars),
                "target_chars_avg": sum(target_chars) / len(target_chars),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
