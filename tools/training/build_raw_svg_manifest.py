from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import boto3


BUCKET = "svg-finetuning-data-446224796301"
INPUT_PREFIX = "dry-run/captioned_arxiv_199"
OUTPUT_PREFIX = "dry-run/raw_svg_arxiv_short150"


def _session(profile: str | None):
    return boto3.Session(profile_name=profile) if profile else boto3.Session()


def _read_s3_lines(s3, bucket: str, key: str) -> list[dict[str, Any]]:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def _valid_svg(svg: str) -> bool:
    if not svg.strip().startswith("<svg"):
        return False
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return False
    return root.tag.endswith("svg")


def _to_raw_svg_record(record: dict[str, Any]) -> dict[str, Any] | None:
    svg = str(record.get("svg") or record.get("completion") or "").strip()
    prompt = str(record.get("prompt") or "").strip()
    if not prompt or not _valid_svg(svg):
        return None

    metadata = dict(record.get("metadata") or {})
    metadata.pop("diagram_ir", None)
    metadata.pop("ir_s3_uri", None)
    metadata["target_format"] = "raw_svg"
    metadata["svg_chars"] = len(svg)

    return {
        "id": str(record.get("id") or hashlib.md5(svg.encode()).hexdigest()[:16]),
        "prompt": prompt,
        "svg": svg,
        "source": str(record.get("source") or "arxiv"),
        "split": "",
        "metadata": metadata,
    }


def _split(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for record in records:
        h = int(hashlib.md5(record["id"].encode()).hexdigest(), 16) % 100
        record["split"] = "val" if h < 10 else "train"
        (val if record["split"] == "val" else train).append(record)
    if not val and len(train) > 1:
        val.append(train.pop())
        val[-1]["split"] = "val"
    return train, val


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records)


def _manifest(records: list[dict[str, Any]], train_uri: str, val_uri: str, limit: int) -> dict[str, Any]:
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    svg_lengths = [len(record["svg"]) for record in records]
    return {
        "schema_version": "1.0",
        "dataset_id": f"raw_svg_arxiv_short{len(records)}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        "created_at": created_at,
        "record_count": len(records),
        "files": [train_uri, val_uri],
        "split": "train",
        "metadata": {
            "source": "captioned_arxiv_raw_svg",
            "input_prefix": f"s3://{BUCKET}/{INPUT_PREFIX}/",
            "target_format": "raw_svg",
            "selection": f"{limit} shortest valid SVG records",
            "min_svg_chars": min(svg_lengths) if svg_lengths else 0,
            "max_svg_chars": max(svg_lengths) if svg_lengths else 0,
            "avg_svg_chars": round(sum(svg_lengths) / len(svg_lengths), 2) if svg_lengths else 0,
        },
        "num_train": sum(1 for record in records if record["split"] == "train"),
        "num_val": sum(1 for record in records if record["split"] == "val"),
        "sources": {"arxiv": len(records)},
        "train_s3_uri": train_uri,
        "val_s3_uri": val_uri,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="svg-finetuning")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--input-prefix", default=INPUT_PREFIX)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--output-dir", default="pipeline_output/raw_svg_arxiv_short150")
    parser.add_argument("--limit", type=int, default=150)
    args = parser.parse_args()

    s3 = _session(args.profile).client("s3")
    raw_rows = []
    raw_rows.extend(_read_s3_lines(s3, args.bucket, f"{args.input_prefix}/train.jsonl"))
    raw_rows.extend(_read_s3_lines(s3, args.bucket, f"{args.input_prefix}/val.jsonl"))

    records = [record for row in raw_rows if (record := _to_raw_svg_record(row)) is not None]
    records = sorted(records, key=lambda record: (len(record["svg"]), record["id"]))
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("no valid raw SVG records found")

    train, val = _split(records)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    manifest_path = out_dir / "manifest.json"
    train_path.write_text(_jsonl(train))
    val_path.write_text(_jsonl(val))

    prefix = args.output_prefix.strip("/")
    train_key = f"{prefix}/train.jsonl"
    val_key = f"{prefix}/val.jsonl"
    manifest_key = f"{prefix}/manifest.json"
    train_uri = f"s3://{args.bucket}/{train_key}"
    val_uri = f"s3://{args.bucket}/{val_key}"
    manifest = _manifest(records, train_uri, val_uri, args.limit)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    s3.put_object(Bucket=args.bucket, Key=train_key, Body=train_path.read_bytes(), ContentType="application/jsonl")
    s3.put_object(Bucket=args.bucket, Key=val_key, Body=val_path.read_bytes(), ContentType="application/jsonl")
    s3.put_object(Bucket=args.bucket, Key=manifest_key, Body=manifest_path.read_bytes(), ContentType="application/json")

    print(json.dumps({"records": len(records), "train": len(train), "val": len(val), "manifest": f"s3://{args.bucket}/{manifest_key}", "metadata": manifest["metadata"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
