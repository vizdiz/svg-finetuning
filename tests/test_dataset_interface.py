from __future__ import annotations

import json
from types import SimpleNamespace

from backend.training.dataset_interface import DatasetLoader, DatasetManifest, TrainingRecord


class _Body:
    def __init__(self, lines: list[dict]):
        self._lines = lines

    def iter_lines(self):
        for line in self._lines:
            yield json.dumps(line).encode()


class _S3:
    def __init__(self, payloads: dict[str, list[dict]]):
        self.payloads = payloads

    def get_object(self, Bucket, Key):
        return {"Body": _Body(self.payloads[f"s3://{Bucket}/{Key}"])}


def test_training_record_prefers_diagram_ir_target_text():
    record = TrainingRecord(
        prompt="draw it",
        svg="<svg/>",
        diagram_ir={"nodes": [{"id": "a"}], "edges": []},
    )

    assert record.target_text() == json.dumps({"edges": [], "nodes": [{"id": "a"}]}, indent=2, sort_keys=True)


def test_dataset_loader_reads_legacy_and_ir_records(monkeypatch):
    manifest = DatasetManifest(
        dataset_id="batch",
        created_at="2026-01-01T00:00:00Z",
        record_count=2,
        files=["s3://bucket/train/dataset.jsonl"],
    )
    payloads = {
        "s3://bucket/train/dataset.jsonl": [
            {"prompt": "legacy", "completion": "<svg id='legacy'/>", "metadata": {"source": "svg"}},
            {"prompt": "ir", "svg": "<svg id='ir'/>", "diagram_ir": {"nodes": [], "edges": []}},
        ]
    }

    loader = DatasetLoader(manifest, s3_client=_S3(payloads))
    records = list(loader.iter_records())

    assert records[0].prompt == "legacy"
    assert records[0].svg == "<svg id='legacy'/>"
    assert records[0].diagram_ir is None
    assert records[0].metadata == {"source": "svg"}
    assert records[1].prompt == "ir"
    assert records[1].svg == "<svg id='ir'/>"
    assert records[1].diagram_ir == {"nodes": [], "edges": []}

