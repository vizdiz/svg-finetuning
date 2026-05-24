from __future__ import annotations

import json
from pathlib import Path

from backend.dataset_pipeline.corpus.schema import read_candidates
from backend.dataset_pipeline.corpus.synthetic_strict_ir import build_synthetic_records, write_synthetic_corpus
from backend.training.diagram_compiler import compile_diagram_ir
from backend.training.diagram_ir import DiagramIRDocument


def test_build_synthetic_records_are_compilable_and_rich_enough():
    records = build_synthetic_records(count=25, seed=11)

    assert len(records) == 25
    assert len({record["id"] for record in records}) == 25
    assert {record["source"] for record in records} == {"synthetic"}
    for record in records:
        doc = DiagramIRDocument.from_dict(record["diagram_ir"])
        svg = compile_diagram_ir(doc)
        assert svg.startswith("<svg")
        assert len(doc.nodes) >= 4
        assert len(doc.edges) >= len(doc.nodes) - 1
        assert "svg" not in record
        assert record["metadata"]["strict_schema"] is True
        assert record["metadata"]["expected_labels"]
        assert record["metadata"]["expected_edges"]


def test_write_synthetic_corpus_outputs_candidates_and_training_manifest(tmp_path):
    result = write_synthetic_corpus(
        corpus_id="unit_synthetic",
        output_root=tmp_path,
        count=30,
        seed=13,
    )

    candidates = read_candidates(Path(result["candidates"]))
    manifest = json.loads(Path(result["manifest"]).read_text())
    training_manifest = json.loads(Path(result["training_manifest"]).read_text())
    train_rows = (tmp_path / "unit_synthetic" / "training" / "train.jsonl").read_text().splitlines()
    val_rows = (tmp_path / "unit_synthetic" / "training" / "val.jsonl").read_text().splitlines()

    assert len(candidates) == 30
    assert manifest["record_count"] == 30
    assert manifest["source_counts"] == {"synthetic": 30}
    assert training_manifest["record_count"] == 30
    assert len(train_rows) + len(val_rows) == 30
