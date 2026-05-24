from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import boto3
from transformers import AutoTokenizer


BUCKET = "svg-finetuning-data-446224796301"
SYSTEM_PROMPT_IR = "You are a diagram compiler. Given a description, produce valid diagram IR JSON only."


def _read_records(input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in ("train.jsonl", "val.jsonl"):
        for line in (input_dir / name).read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _compact_target(record: dict[str, Any]) -> str:
    diagram_ir = record.get("diagram_ir")
    if diagram_ir is None:
        diagram_ir = (record.get("metadata") or {}).get("diagram_ir")
    if diagram_ir is None:
        raise ValueError(f"Record {record.get('id') or '<unknown>'} has no diagram_ir")
    return json.dumps(diagram_ir, sort_keys=True, separators=(",", ":"))


def _full_token_count(tokenizer, prompt: str, target: str) -> int:
    return len(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT_IR},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )
    )


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def _assign_split(record: dict[str, Any]) -> str:
    record_id = str(record.get("id") or "")
    h = int(hashlib.md5(record_id.encode()).hexdigest(), 16) % 100
    return "val" if h < 10 else "train"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="pipeline_output/canonical_arxiv_short150")
    parser.add_argument("--output-dir", default="pipeline_output/canonical_arxiv_full_ir_fit4096")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--s3-prefix", default="dry-run/canonical_arxiv_full_ir_fit4096")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--profile", default="svg-finetuning")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for record in _read_records(input_dir):
        target = _compact_target(record)
        token_count = _full_token_count(tokenizer, record["prompt"], target)
        metadata = dict(record.get("metadata") or {})
        metadata["full_ir_training_tokens"] = token_count
        metadata["full_ir_target_chars"] = len(target)
        record["metadata"] = metadata
        if token_count <= args.max_length:
            record["split"] = _assign_split(record)
            kept.append(record)
        else:
            skipped.append(
                {
                    "id": record.get("id", ""),
                    "source": record.get("source", ""),
                    "tokens": token_count,
                    "target_chars": len(target),
                }
            )

    if not kept:
        raise ValueError(f"No records fit max_length={args.max_length}")

    train_records = [record for record in kept if record.get("split") == "train"]
    val_records = [record for record in kept if record.get("split") == "val"]
    if not val_records and len(train_records) > 1:
        val_records = [train_records.pop()]
        val_records[0]["split"] = "val"

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    skipped_path = output_dir / "skipped_overlength.jsonl"
    manifest_path = output_dir / "manifest.json"
    train_path.write_text(_jsonl(train_records))
    val_path.write_text(_jsonl(val_records))
    skipped_path.write_text(_jsonl(skipped))

    s3 = boto3.Session(profile_name=args.profile).client("s3")
    prefix = args.s3_prefix.strip("/")
    train_key = f"{prefix}/train.jsonl"
    val_key = f"{prefix}/val.jsonl"
    manifest_key = f"{prefix}/manifest.json"
    train_uri = f"s3://{args.bucket}/{train_key}"
    val_uri = f"s3://{args.bucket}/{val_key}"
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest = {
        "schema_version": "1.0",
        "dataset_id": f"canonical_arxiv_full_ir_fit4096_{len(kept)}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        "created_at": created_at,
        "record_count": len(kept),
        "files": [train_uri, val_uri],
        "split": "train",
        "metadata": {
            "source": "canonical_arxiv_full_ir_fit",
            "target_format": "diagram_ir",
            "full_ir_only": True,
            "max_length": args.max_length,
            "skipped_overlength": len(skipped),
        },
        "num_train": len(train_records),
        "num_val": len(val_records),
        "train_s3_uri": train_uri,
        "val_s3_uri": val_uri,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    s3.put_object(Bucket=args.bucket, Key=train_key, Body=train_path.read_bytes(), ContentType="application/jsonl")
    s3.put_object(Bucket=args.bucket, Key=val_key, Body=val_path.read_bytes(), ContentType="application/jsonl")
    s3.put_object(Bucket=args.bucket, Key=manifest_key, Body=manifest_path.read_bytes(), ContentType="application/json")

    token_counts = [record["metadata"]["full_ir_training_tokens"] for record in kept]
    print(
        json.dumps(
            {
                "records_in": len(kept) + len(skipped),
                "kept": len(kept),
                "skipped": len(skipped),
                "train": len(train_records),
                "val": len(val_records),
                "tokens_max": max(token_counts),
                "tokens_avg": sum(token_counts) / len(token_counts),
                "manifest": f"s3://{args.bucket}/{manifest_key}",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
