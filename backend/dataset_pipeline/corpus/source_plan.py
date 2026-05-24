from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from backend.dataset_pipeline.corpus.schema import VALID_SOURCES, utc_now


DEFAULT_SOURCE_WEIGHTS = {
    "synthetic": 0.20,
    "arxiv": 0.40,
    "wikimedia": 0.15,
    "github": 0.15,
    "commoncrawl": 0.10,
}

BULK_SOURCE_METHODS = {
    "synthetic": "local_generator",
    "arxiv": "s3_requester_pays_source_tarballs",
    "wikimedia": "sql_xml_dumps_then_selective_original_fetch",
    "github": "public_dataset_or_bigquery_candidate_selection",
    "commoncrawl": "index_wat_warc_bulk_records",
}


@dataclass(slots=True)
class SourceAllocation:
    source: str
    target_records: int
    method: str
    target_format: str
    notes: str = ""

    def validate(self) -> None:
        if self.source not in VALID_SOURCES:
            raise ValueError(f"unknown source: {self.source}")
        if self.target_records < 0:
            raise ValueError("target_records must be non-negative")
        if not self.method:
            raise ValueError("method is required")
        if not self.target_format:
            raise ValueError("target_format is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class CorpusBuildPlan:
    plan_id: str
    target_records: int
    created_at: str
    allocations: list[SourceAllocation] = field(default_factory=list)
    quality_gates: dict[str, Any] = field(default_factory=dict)
    storage: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.plan_id:
            raise ValueError("plan_id is required")
        if self.target_records <= 0:
            raise ValueError("target_records must be positive")
        for allocation in self.allocations:
            allocation.validate()
        allocated = sum(item.target_records for item in self.allocations)
        if allocated != self.target_records:
            raise ValueError(f"allocations sum to {allocated}, expected {self.target_records}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["allocations"] = [allocation.to_dict() for allocation in self.allocations]
        return payload


def _rounded_allocations(target_records: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {source: target_records * weight for source, weight in weights.items()}
    allocations = {source: int(value) for source, value in raw.items()}
    remainder = target_records - sum(allocations.values())
    fractional_order = sorted(raw, key=lambda source: raw[source] - int(raw[source]), reverse=True)
    for source in fractional_order[:remainder]:
        allocations[source] += 1
    return allocations


def build_default_plan(target_records: int = 50000, *, plan_id: str | None = None) -> CorpusBuildPlan:
    if target_records <= 0:
        raise ValueError("target_records must be positive")
    allocations = _rounded_allocations(target_records, DEFAULT_SOURCE_WEIGHTS)
    source_allocations = [
        SourceAllocation(
            source="synthetic",
            target_records=allocations["synthetic"],
            method=BULK_SOURCE_METHODS["synthetic"],
            target_format="diagram_ir",
            notes="Clean schema curriculum data generated locally; no network.",
        ),
        SourceAllocation(
            source="arxiv",
            target_records=allocations["arxiv"],
            method=BULK_SOURCE_METHODS["arxiv"],
            target_format="diagram_ir",
            notes="Use requester-pays S3 source tarballs; extract TeX/TikZ/SVG/PDF figure artifacts.",
        ),
        SourceAllocation(
            source="wikimedia",
            target_records=allocations["wikimedia"],
            method=BULK_SOURCE_METHODS["wikimedia"],
            target_format="diagram_ir",
            notes="Rank from dumps first; fetch only selected SVG originals with cache/provenance.",
        ),
        SourceAllocation(
            source="github",
            target_records=allocations["github"],
            method=BULK_SOURCE_METHODS["github"],
            target_format="raw_svg",
            notes="Use public datasets for candidate selection; fetch selected raw blobs only.",
        ),
        SourceAllocation(
            source="commoncrawl",
            target_records=allocations["commoncrawl"],
            method=BULK_SOURCE_METHODS["commoncrawl"],
            target_format="raw_svg",
            notes="Read WAT/WARC records; do not request origin sites.",
        ),
    ]
    return CorpusBuildPlan(
        plan_id=plan_id or f"corpus_plan_{target_records}",
        target_records=target_records,
        created_at=utc_now(),
        allocations=source_allocations,
        quality_gates={
            "dedupe": ["content_sha256", "normalized_svg_hash", "near_duplicate_render_hash"],
            "svg_hard_rejects": ["invalid_xml", "script_tag", "external_reference", "missing_viewbox_or_dimensions"],
            "ir_hard_rejects": ["schema_invalid", "compile_failed", "edge_unknown_node"],
            "minimums": {
                "node_count": 4,
                "edge_count": 3,
                "label_recall": 0.95,
                "edge_recall": 0.90,
                "compilability": 1.0,
            },
        },
        storage={
            "local_root": "pipeline_output/corpus",
            "s3_root": "s3://svg-finetuning-data-446224796301/corpus",
            "write_order": ["candidates", "assets", "quality_reports", "training_splits", "manifest"],
            "training_manifest_trigger": "manual_only_until_approved",
        },
    )


def write_plan(path: Path, plan: CorpusBuildPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
