from __future__ import annotations

from tools.training.build_strict_ir_manifest import build_records


def test_strict_ir_records_use_clean_compiler_ir_targets():
    records = build_records(copies=1)

    assert records
    for record in records:
        assert record["prompt"].strip()
        assert "diagram_ir" in record
        assert "svg" not in record
        assert record["metadata"]["strict_schema"] is True
        assert record["metadata"]["target_format"] == "diagram_ir"
        labels = [node["label"] for node in record["diagram_ir"]["nodes"]]
        assert labels == record["metadata"]["expected_labels"]
        assert len(record["diagram_ir"]["edges"]) == len(record["metadata"]["expected_edges"])
