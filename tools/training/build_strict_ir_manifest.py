from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.training.diagram_compiler import compile_diagram_ir
from backend.training.diagram_ir import (
    CanvasSpec,
    DiagramIRDocument,
    EdgeSpec,
    LayoutSpec,
    NodeSpec,
)


BUCKET = "svg-finetuning-data-446224796301"


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")


def _doc(
    *,
    title: str,
    diagram_type: str,
    labels: list[str],
    edges: list[tuple[str, str]],
    direction: str = "horizontal",
    edge_labels: dict[tuple[str, str], str] | None = None,
) -> DiagramIRDocument:
    edge_labels = edge_labels or {}
    nodes = [
        NodeSpec(
            id=_slug(label),
            kind="box",
            label=label,
        )
        for label in labels
    ]
    edge_specs = [
        EdgeSpec(
            id=f"edge-{index}",
            source=_slug(source),
            target=_slug(target),
            directed=True,
            label=edge_labels.get((source, target), ""),
            routing="straight",
        )
        for index, (source, target) in enumerate(edges, start=1)
    ]
    document = DiagramIRDocument(
        diagram_type=diagram_type,
        title=title,
        canvas=CanvasSpec(width=640, height=480, padding=64),
        nodes=nodes,
        edges=edge_specs,
        layout=LayoutSpec(direction=direction, spacing=72, edge_routing="straight"),
        metadata={"dataset": "strict_ir"},
    )
    document.validate()
    compile_diagram_ir(document)
    return document


def _record(
    *,
    record_id: str,
    prompt: str,
    document: DiagramIRDocument,
    source: str = "strict_ir_seed",
) -> dict[str, Any]:
    labels = [node.label for node in document.nodes]
    edges = [
        {
            "source": document.node_map()[edge.source].label,
            "target": document.node_map()[edge.target].label,
            "label": edge.label,
        }
        for edge in document.edges
    ]
    return {
        "id": record_id,
        "prompt": prompt,
        "diagram_ir": document.to_dict(),
        "source": source,
        "split": "",
        "metadata": {
            "target_format": "diagram_ir",
            "strict_schema": True,
            "expected_labels": labels,
            "expected_edges": edges,
        },
    }


def _base_specs() -> list[dict[str, Any]]:
    return [
        {
            "title": "SVG generation pipeline",
            "diagram_type": "architecture",
            "labels": ["browser", "API gateway", "diagram model", "IR compiler", "SVG validator", "download"],
            "edges": [
                ("browser", "API gateway"),
                ("API gateway", "diagram model"),
                ("diagram model", "IR compiler"),
                ("IR compiler", "SVG validator"),
                ("SVG validator", "download"),
            ],
            "prompts": [
                "Create a left-to-right architecture diagram with boxes labeled browser, API gateway, diagram model, IR compiler, SVG validator, and download. Connect them in that order.",
                "Draw the svgen request path from browser to API gateway to diagram model to IR compiler to SVG validator to download.",
            ],
        },
        {
            "title": "JWT auth sequence",
            "diagram_type": "sequence",
            "labels": ["client", "API gateway", "auth service", "token store", "protected API"],
            "edges": [
                ("client", "API gateway"),
                ("API gateway", "auth service"),
                ("auth service", "token store"),
                ("auth service", "API gateway"),
                ("API gateway", "protected API"),
            ],
            "edge_labels": {
                ("client", "API gateway"): "login",
                ("API gateway", "auth service"): "validate",
                ("auth service", "token store"): "issue token",
                ("API gateway", "protected API"): "authorized request",
            },
            "prompts": [
                "Create a JWT auth flow with client, API gateway, auth service, token store, and protected API. Show login, validation, token issuing, and authorized request arrows.",
                "Draw a left-to-right sequence diagram for JWT authentication between client, API gateway, auth service, token store, and protected API.",
            ],
        },
        {
            "title": "ETL pipeline",
            "diagram_type": "flowchart",
            "labels": ["raw events", "queue", "worker", "normalizer", "warehouse", "dashboard"],
            "edges": [
                ("raw events", "queue"),
                ("queue", "worker"),
                ("worker", "normalizer"),
                ("normalizer", "warehouse"),
                ("warehouse", "dashboard"),
            ],
            "prompts": [
                "Draw an ETL pipeline where raw events flow to a queue, worker, normalizer, warehouse, and dashboard.",
                "Create a technical data pipeline diagram with raw events, queue, worker, normalizer, warehouse, and dashboard connected left to right.",
            ],
        },
        {
            "title": "Branching moderation workflow",
            "diagram_type": "flowchart",
            "labels": ["upload", "scanner", "policy check", "approve", "review queue", "reject"],
            "edges": [
                ("upload", "scanner"),
                ("scanner", "policy check"),
                ("policy check", "approve"),
                ("policy check", "review queue"),
                ("review queue", "reject"),
            ],
            "edge_labels": {
                ("policy check", "approve"): "safe",
                ("policy check", "review queue"): "uncertain",
                ("review queue", "reject"): "violation",
            },
            "prompts": [
                "Create a branching content moderation workflow: upload to scanner to policy check, then safe items approve, uncertain items go to review queue, and violations reject.",
                "Draw a moderation flowchart with upload, scanner, policy check, approve, review queue, and reject. Label the branches safe, uncertain, and violation.",
            ],
        },
        {
            "title": "Cache fallback path",
            "diagram_type": "architecture",
            "labels": ["request", "cache", "model endpoint", "compiler", "response"],
            "edges": [
                ("request", "cache"),
                ("cache", "response"),
                ("cache", "model endpoint"),
                ("model endpoint", "compiler"),
                ("compiler", "response"),
            ],
            "edge_labels": {
                ("cache", "response"): "hit",
                ("cache", "model endpoint"): "miss",
            },
            "prompts": [
                "Draw a cache-first generation path: request checks cache, cache hit returns response, cache miss calls model endpoint, then compiler, then response.",
                "Create an architecture diagram for request, cache, model endpoint, compiler, and response with hit and miss paths.",
            ],
        },
    ]


def build_records(copies: int = 6) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(_base_specs(), start=1):
        document = _doc(
            title=spec["title"],
            diagram_type=spec["diagram_type"],
            labels=spec["labels"],
            edges=spec["edges"],
            edge_labels=spec.get("edge_labels"),
        )
        for copy_index in range(copies):
            prompt = spec["prompts"][copy_index % len(spec["prompts"])]
            if copy_index >= len(spec["prompts"]):
                prompt = f"{prompt} Use simple labeled boxes and straight connector arrows."
            record_id = f"strict-ir-{spec_index:02d}-{copy_index + 1:02d}"
            records.append(_record(record_id=record_id, prompt=prompt, document=document))
    return records


def _split(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for record in records:
        h = int(hashlib.md5(record["id"].encode()).hexdigest(), 16) % 100
        record["split"] = "val" if h < 15 else "train"
        (val if record["split"] == "val" else train).append(record)
    return train, val


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="pipeline_output/strict_ir_seed")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--s3-prefix", default="dry-run/strict_ir_seed")
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--profile", default="svg-finetuning")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    records = build_records(copies=args.copies)
    train, val = _split(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.jsonl").write_text(_jsonl(train))
    (output_dir / "val.jsonl").write_text(_jsonl(val))

    prefix = args.s3_prefix.strip("/")
    train_uri = f"s3://{args.bucket}/{prefix}/train.jsonl"
    val_uri = f"s3://{args.bucket}/{prefix}/val.jsonl"
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest = {
        "schema_version": "1.0",
        "dataset_id": f"strict_ir_seed_{len(records)}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        "created_at": created_at,
        "record_count": len(records),
        "files": [train_uri, val_uri],
        "split": "train",
        "metadata": {
            "source": "strict_ir_seed",
            "target_format": "diagram_ir",
            "strict_schema": True,
            "raw_svg_debug_payload": False,
        },
        "num_train": len(train),
        "num_val": len(val),
        "train_s3_uri": train_uri,
        "val_s3_uri": val_uri,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    if not args.no_upload:
        s3 = boto3.Session(profile_name=args.profile).client("s3")
        s3.put_object(Bucket=args.bucket, Key=f"{prefix}/train.jsonl", Body=(output_dir / "train.jsonl").read_bytes(), ContentType="application/jsonl")
        s3.put_object(Bucket=args.bucket, Key=f"{prefix}/val.jsonl", Body=(output_dir / "val.jsonl").read_bytes(), ContentType="application/jsonl")
        s3.put_object(Bucket=args.bucket, Key=f"{prefix}/manifest.json", Body=(output_dir / "manifest.json").read_bytes(), ContentType="application/json")

    print(json.dumps({"records": len(records), "train": len(train), "val": len(val), "manifest": f"s3://{args.bucket}/{prefix}/manifest.json"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
