from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from backend.dataset_pipeline.corpus.schema import utc_now, write_jsonl


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _split(rows: list[dict[str, Any]], val_percent: int = 10) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for row in rows:
        try:
            bucket = int(str(row.get("id", "0"))[-8:], 16) % 100 if row.get("id") else 0
        except ValueError:
            bucket = sum(ord(ch) for ch in str(row.get("id", ""))) % 100
        row = dict(row)
        row["split"] = "val" if bucket < val_percent else "train"
        (val if row["split"] == "val" else train).append(row)
    return train, val


def assemble_dataset_products(
    *,
    dataset_id: str,
    normalized_record_files: list[Path],
    output_root: Path,
    val_percent: int = 10,
) -> dict[str, Any]:
    rows = [row for path in normalized_record_files for row in _read_jsonl(path)]
    products = {
        "strict_ir_train": [row for row in rows if row.get("diagram_ir") is not None],
        "raw_svg_train": [row for row in rows if row.get("svg")],
        "svg_to_ir_train": [row for row in rows if row.get("svg") and row.get("diagram_ir") is not None],
    }

    root = output_root / dataset_id
    root.mkdir(parents=True, exist_ok=True)
    product_manifests: dict[str, dict[str, Any]] = {}
    for name, product_rows in products.items():
        product_dir = root / name
        train, val = _split(product_rows, val_percent=val_percent)
        write_jsonl(product_dir / "train.jsonl", train)
        write_jsonl(product_dir / "val.jsonl", val)
        manifest = {
            "schema_version": "1.0",
            "dataset_id": f"{dataset_id}_{name}",
            "created_at": utc_now(),
            "record_count": len(product_rows),
            "files": [str(product_dir / "train.jsonl"), str(product_dir / "val.jsonl")],
            "split": "train",
            "metadata": {"product": name, "num_train": len(train), "num_val": len(val)},
        }
        (product_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        product_manifests[name] = manifest

    eval_contract = [
        {
            "id": row.get("id", ""),
            "prompt": row.get("prompt", ""),
            "expected_labels": row.get("metadata", {}).get("expected_labels", []),
            "expected_edges": row.get("metadata", {}).get("expected_edges", []),
        }
        for row in products["strict_ir_train"]
        if row.get("metadata", {}).get("expected_labels")
    ]
    write_jsonl(root / "eval_contract" / "records.jsonl", eval_contract)
    visual_rows = [{"id": row.get("id", ""), "source": row.get("source", ""), "svg": row.get("svg", "")} for row in products["raw_svg_train"][:500]]
    write_jsonl(root / "eval_visual" / "records.jsonl", visual_rows)

    summary = {
        "dataset_id": dataset_id,
        "created_at": utc_now(),
        "input_files": [str(path) for path in normalized_record_files],
        "products": {name: {"record_count": manifest["record_count"], **manifest["metadata"]} for name, manifest in product_manifests.items()},
        "eval_contract_count": len(eval_contract),
        "eval_visual_count": len(visual_rows),
    }
    (root / "assembly_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return {"dataset_id": dataset_id, "root": str(root), "summary": str(root / "assembly_summary.json"), "products": summary["products"]}
