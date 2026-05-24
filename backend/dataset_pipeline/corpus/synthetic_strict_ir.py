from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import random
from typing import Any

from backend.dataset_pipeline.corpus.schema import (
    CorpusCandidate,
    build_manifest,
    local_corpus_paths,
    s3_corpus_prefix,
    stable_id,
    utc_now,
    write_jsonl,
)
from backend.training.diagram_compiler import compile_diagram_ir
from backend.training.diagram_ir import CanvasSpec, DiagramIRDocument, EdgeSpec, LayoutSpec, NodeSpec


DOMAINS: dict[str, dict[str, Any]] = {
    "auth": {
        "diagram_types": ["sequence", "flowchart"],
        "entities": ["user", "web app", "API gateway", "auth service", "OAuth provider", "token service", "token store", "protected API", "audit log"],
        "edge_labels": ["login", "redirect", "authorize", "callback", "exchange code", "issue token", "validate token", "authorized request", "log event"],
        "verbs": ["auth flow", "OAuth flow", "JWT authentication flow", "login sequence"],
    },
    "data": {
        "diagram_types": ["flowchart", "architecture"],
        "entities": ["event source", "ingest API", "queue", "stream processor", "normalizer", "feature store", "warehouse", "dashboard", "alert service"],
        "edge_labels": ["publish", "enqueue", "consume", "normalize", "write", "aggregate", "read", "notify"],
        "verbs": ["ETL pipeline", "analytics pipeline", "streaming data flow", "event processing workflow"],
    },
    "ml": {
        "diagram_types": ["architecture", "flowchart"],
        "entities": ["user prompt", "API gateway", "captioner", "fine-tuned model", "IR repair", "compiler", "validator", "cache", "SVG response"],
        "edge_labels": ["submit", "describe media", "generate IR", "repair", "compile", "validate", "store", "return"],
        "verbs": ["model inference pipeline", "SVG generation workflow", "diagram generation architecture", "multimodal generation flow"],
    },
    "moderation": {
        "diagram_types": ["flowchart"],
        "entities": ["upload", "scanner", "policy check", "risk scorer", "approve", "review queue", "human reviewer", "reject", "appeal"],
        "edge_labels": ["scan", "classify", "score", "safe", "uncertain", "review", "violation", "appeal"],
        "verbs": ["content moderation workflow", "policy review flow", "safety triage pipeline", "upload moderation process"],
    },
    "commerce": {
        "diagram_types": ["sequence", "flowchart"],
        "entities": ["shopper", "storefront", "cart service", "payment gateway", "fraud check", "inventory", "order service", "email service"],
        "edge_labels": ["add item", "checkout", "authorize", "screen", "reserve stock", "create order", "send receipt"],
        "verbs": ["checkout flow", "commerce order workflow", "payment authorization sequence", "storefront purchase flow"],
    },
    "infra": {
        "diagram_types": ["architecture"],
        "entities": ["browser", "CDN", "load balancer", "web service", "worker", "database", "object store", "metrics", "alerting"],
        "edge_labels": ["request", "route", "serve", "enqueue", "query", "store", "emit metrics", "alert"],
        "verbs": ["web service architecture", "cloud deployment diagram", "request handling path", "production service topology"],
    },
}

LAYOUTS = ["horizontal", "vertical"]


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")


def _choose_entities(rng: random.Random, domain: dict[str, Any], min_nodes: int, max_nodes: int) -> list[str]:
    count = rng.randint(min_nodes, max_nodes)
    entities = list(domain["entities"])
    rng.shuffle(entities)
    selected = entities[:count]
    if rng.random() < 0.35 and count >= 5:
        # Keep some workflows naturally ordered when the domain list encodes a common path.
        selected = domain["entities"][:count]
    return selected


def _build_edges(
    rng: random.Random,
    entities: list[str],
    edge_labels: list[str],
    *,
    branch_probability: float = 0.25,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], str]]:
    edges: list[tuple[str, str]] = []
    labels: dict[tuple[str, str], str] = {}
    for index, (source, target) in enumerate(zip(entities, entities[1:])):
        edge = (source, target)
        edges.append(edge)
        labels[edge] = edge_labels[index % len(edge_labels)] if rng.random() < 0.7 else ""

    if len(entities) >= 5 and rng.random() < branch_probability:
        source_index = rng.randint(1, len(entities) - 3)
        target_index = rng.randint(source_index + 2, len(entities) - 1)
        edge = (entities[source_index], entities[target_index])
        if edge not in edges:
            edges.append(edge)
            labels[edge] = rng.choice(edge_labels)

    if len(entities) >= 4 and rng.random() < 0.2:
        edge = (entities[-2], entities[1])
        if edge not in edges:
            edges.append(edge)
            labels[edge] = rng.choice(["retry", "refresh", "callback", "feedback"])

    return edges, labels


def build_document(
    *,
    title: str,
    diagram_type: str,
    entities: list[str],
    edges: list[tuple[str, str]],
    edge_labels: dict[tuple[str, str], str],
    direction: str,
) -> DiagramIRDocument:
    nodes = [NodeSpec(id=_slug(entity), kind="box", label=entity) for entity in entities]
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
        groups=[],
        layout=LayoutSpec(direction=direction, spacing=72, edge_routing="straight"),
        metadata={"dataset": "synthetic_strict_ir"},
    )
    document.validate()
    compile_diagram_ir(document)
    return document


def _prompt(
    rng: random.Random,
    *,
    verb: str,
    diagram_type: str,
    entities: list[str],
    edges: list[tuple[str, str]],
    direction: str,
) -> str:
    entity_text = ", ".join(entities)
    if rng.random() < 0.5:
        edge_text = "; ".join(f"{source} to {target}" for source, target in edges[:8])
        return (
            f"Create a {direction} {diagram_type} for a {verb}. "
            f"Include boxes labeled {entity_text}. Connect {edge_text}."
        )
    return (
        f"Draw a {verb} as a {diagram_type}. Use {direction} layout, simple labeled boxes, "
        f"and straight connector arrows. Required labels: {entity_text}."
    )


def build_synthetic_records(
    *,
    count: int,
    seed: int = 7,
    min_nodes: int = 4,
    max_nodes: int = 8,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    domains = list(DOMAINS)
    records: list[dict[str, Any]] = []
    for index in range(count):
        domain_name = domains[index % len(domains)]
        domain = DOMAINS[domain_name]
        entities = _choose_entities(rng, domain, min_nodes, max_nodes)
        edges, edge_labels = _build_edges(rng, entities, domain["edge_labels"])
        diagram_type = rng.choice(domain["diagram_types"])
        direction = rng.choice(LAYOUTS)
        verb = rng.choice(domain["verbs"])
        title = f"{verb.title()} {index + 1}"
        document = build_document(
            title=title,
            diagram_type=diagram_type,
            entities=entities,
            edges=edges,
            edge_labels=edge_labels,
            direction=direction,
        )
        prompt = _prompt(
            rng,
            verb=verb,
            diagram_type=diagram_type,
            entities=entities,
            edges=edges,
            direction=direction,
        )
        expected_edges = [
            {"source": source, "target": target, "label": edge_labels.get((source, target), "")}
            for source, target in edges
        ]
        record_id = stable_id("synthetic", seed, index, prompt, prefix="strict-ir")
        records.append(
            {
                "id": record_id,
                "prompt": prompt,
                "diagram_ir": document.to_dict(),
                "source": "synthetic",
                "split": "",
                "metadata": {
                    "target_format": "diagram_ir",
                    "strict_schema": True,
                    "domain": domain_name,
                    "expected_labels": entities,
                    "expected_edges": expected_edges,
                    "synthetic_seed": seed,
                    "synthetic_index": index,
                },
            }
        )
    return records


def records_to_candidates(records: list[dict[str, Any]], *, corpus_id: str) -> list[CorpusCandidate]:
    candidates: list[CorpusCandidate] = []
    for record in records:
        target = json.dumps(record["diagram_ir"], sort_keys=True, separators=(",", ":"))
        candidates.append(
            CorpusCandidate(
                candidate_id=record["id"],
                source="synthetic",
                uri=f"synthetic://{corpus_id}/{record['id']}",
                target_format="diagram_ir",
                title=record["diagram_ir"].get("title", ""),
                caption=record["prompt"],
                license="internal-synthetic",
                provenance={
                    "generator": "synthetic_strict_ir",
                    "created_at": utc_now(),
                },
                metadata={
                    **record["metadata"],
                    "target_chars": len(target),
                    "node_count": len(record["diagram_ir"].get("nodes", [])),
                    "edge_count": len(record["diagram_ir"].get("edges", [])),
                },
                priority_score=1.0,
                fetch_status="not_required",
            )
        )
    return candidates


def split_records(records: list[dict[str, Any]], *, val_percent: int = 10) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for record in records:
        bucket = int(record["id"].rsplit("-", 1)[-1], 16) % 100
        record["split"] = "val" if bucket < val_percent else "train"
        (val if record["split"] == "val" else train).append(record)
    return train, val


def write_synthetic_corpus(
    *,
    corpus_id: str,
    output_root: Path,
    count: int,
    seed: int = 7,
    upload_bucket: str = "",
    s3_client: Any | None = None,
) -> dict[str, Any]:
    records = build_synthetic_records(count=count, seed=seed)
    candidates = records_to_candidates(records, corpus_id=corpus_id)
    paths = local_corpus_paths(output_root, corpus_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["candidates"], [candidate.to_dict() for candidate in candidates])

    train, val = split_records(records)
    training_dir = paths["training"]
    training_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(training_dir / "train.jsonl", train)
    write_jsonl(training_dir / "val.jsonl", val)
    training_manifest = {
        "schema_version": "1.0",
        "dataset_id": corpus_id,
        "created_at": utc_now(),
        "record_count": len(records),
        "files": [str(training_dir / "train.jsonl"), str(training_dir / "val.jsonl")],
        "split": "train",
        "metadata": {
            "source": "synthetic_strict_ir",
            "target_format": "diagram_ir",
            "strict_schema": True,
        },
        "num_train": len(train),
        "num_val": len(val),
    }
    (training_dir / "manifest.json").write_text(json.dumps(training_manifest, indent=2, sort_keys=True))

    candidates_uri = str(paths["candidates"])
    if upload_bucket:
        import boto3

        s3_client = s3_client or boto3.client("s3")
        prefix = s3_corpus_prefix(corpus_id)
        for local_path, key_suffix, content_type in (
            (paths["candidates"], "candidates.jsonl", "application/jsonl"),
            (training_dir / "train.jsonl", "training/train.jsonl", "application/jsonl"),
            (training_dir / "val.jsonl", "training/val.jsonl", "application/jsonl"),
            (training_dir / "manifest.json", "training/manifest.json", "application/json"),
        ):
            s3_client.put_object(
                Bucket=upload_bucket,
                Key=f"{prefix}/{key_suffix}",
                Body=local_path.read_bytes(),
                ContentType=content_type,
            )
        candidates_uri = f"s3://{upload_bucket}/{prefix}/candidates.jsonl"

    manifest = build_manifest(
        corpus_id,
        candidates_uri,
        candidates,
        metadata={
            "generator": "synthetic_strict_ir",
            "seed": seed,
            "training_manifest": str(training_dir / "manifest.json"),
            "s3_prefix": s3_corpus_prefix(corpus_id),
        },
    )
    paths["manifest"].write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    if upload_bucket:
        s3_client.put_object(
            Bucket=upload_bucket,
            Key=f"{s3_corpus_prefix(corpus_id)}/manifest.json",
            Body=paths["manifest"].read_bytes(),
            ContentType="application/json",
        )

    return {
        "corpus_id": corpus_id,
        "records": len(records),
        "train": len(train),
        "val": len(val),
        "manifest": str(paths["manifest"]),
        "candidates": str(paths["candidates"]),
        "training_manifest": str(training_dir / "manifest.json"),
    }
