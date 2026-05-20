"""
Unit tests for dataset_pipeline.pipeline.manifest_writer.

Run with:
    python -m pytest tests/manifest_writer.py -v
"""

import hashlib
import json
import logging
from dataclasses import asdict
from types import SimpleNamespace

from backend.dataset_pipeline.pipeline import manifest_writer
from backend.dataset_pipeline.processing.captioner import TrainingRecord


def _split_for(record_id: str) -> str:
    h = int(hashlib.md5(record_id.encode()).hexdigest(), 16) % 100
    return "val" if h < 10 else "train"


def _id_for_split(split: str) -> str:
    for i in range(1000):
        record_id = f"{split}-{i}"
        if _split_for(record_id) == split:
            return record_id
    raise AssertionError(f"Could not find id for split {split}")


def _record(record_id: str, source: str) -> TrainingRecord:
    return TrainingRecord(
        id=record_id,
        prompt=f"prompt {record_id}",
        completion=f"<svg id='{record_id}'/>",
        source=source,
        split="",
        metadata={"source_id": record_id},
    )


def _records() -> list[TrainingRecord]:
    return [
        _record(_id_for_split("train"), "arxiv"),
        _record(_id_for_split("val"), "wikipedia"),
        _record("extra-arxiv", "arxiv"),
    ]


def _jsonl_records(body: bytes) -> list[dict]:
    text = body.decode()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


class _FakeS3:
    def __init__(self):
        self.puts = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)


def test_write_manifest_uploads_jsonl_then_compatible_manifest(monkeypatch):
    records = _records()
    fake_s3 = _FakeS3()
    monkeypatch.setattr(manifest_writer.boto3, "client", lambda name: fake_s3)

    uri = manifest_writer.write_manifest(records, SimpleNamespace(s3_data_bucket="bucket"))

    assert uri == "s3://bucket/train/dataset_manifest.json"
    assert [put["Key"] for put in fake_s3.puts] == [
        "train/dataset.jsonl",
        "val/dataset.jsonl",
        "train/dataset_manifest.json",
    ]
    assert fake_s3.puts[-1]["ContentType"] == "application/json"

    train_lines = _jsonl_records(fake_s3.puts[0]["Body"])
    val_lines = _jsonl_records(fake_s3.puts[1]["Body"])
    manifest = json.loads(fake_s3.puts[2]["Body"])

    assert train_lines == [asdict(record) for record in records if record.split == "train"]
    assert val_lines == [asdict(record) for record in records if record.split == "val"]
    assert manifest["train_s3_uri"] == "s3://bucket/train/dataset.jsonl"
    assert manifest["val_s3_uri"] == "s3://bucket/val/dataset.jsonl"
    assert manifest["files"] == [manifest["train_s3_uri"], manifest["val_s3_uri"]]
    assert manifest["dataset_id"] == manifest["version"]
    assert manifest["record_count"] == manifest["total"] == len(records)
    assert manifest["num_train"] == len(train_lines)
    assert manifest["num_val"] == len(val_lines)
    assert manifest["sources"] == {"arxiv": 2, "wikipedia": 1}


def test_write_manifest_assigns_splits_in_place(monkeypatch):
    records = _records()
    fake_s3 = _FakeS3()
    monkeypatch.setattr(manifest_writer.boto3, "client", lambda name: fake_s3)

    manifest_writer.write_manifest(records, SimpleNamespace(s3_data_bucket="bucket"))

    assert [record.split for record in records] == [_split_for(record.id) for record in records]
    assert {record.split for record in records} == {"train", "val"}


def test_dry_run_writes_local_files_without_trigger_manifest(tmp_path):
    records = _records()

    manifest_path = tmp_path / "manifest.json"
    returned = manifest_writer.dry_run_write(records, str(tmp_path))

    assert returned == str(manifest_path)
    assert manifest_path.exists()
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert not (tmp_path / "dataset_manifest.json").exists()

    train_lines = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    val_lines = [json.loads(line) for line in (tmp_path / "val.jsonl").read_text().splitlines()]
    manifest = json.loads(manifest_path.read_text())

    assert train_lines == [asdict(record) for record in records if record.split == "train"]
    assert val_lines == [asdict(record) for record in records if record.split == "val"]
    assert manifest["files"] == [str(tmp_path / "train.jsonl"), str(tmp_path / "val.jsonl")]
    assert manifest["dataset_id"] == manifest["version"]
    assert manifest["record_count"] == manifest["total"] == len(records)


def test_dry_run_can_upload_artifacts_to_s3(monkeypatch, tmp_path):
    records = _records()
    fake_s3 = _FakeS3()
    monkeypatch.setattr(manifest_writer.boto3, "client", lambda name: fake_s3)

    returned = manifest_writer.dry_run_write(
        records,
        str(tmp_path),
        config=SimpleNamespace(s3_data_bucket="bucket"),
        upload_to_s3=True,
        dry_run_s3_prefix="dry-run/test-run",
    )

    assert returned == f"s3://bucket/dry-run/test-run/{tmp_path.name}/manifest.json"
    assert [put["Key"] for put in fake_s3.puts] == [
        f"dry-run/test-run/{tmp_path.name}/train.jsonl",
        f"dry-run/test-run/{tmp_path.name}/val.jsonl",
        f"dry-run/test-run/{tmp_path.name}/manifest.json",
    ]
    assert fake_s3.puts[-1]["ContentType"] == "application/json"


def test_summary_table_is_logged(caplog, tmp_path):
    caplog.set_level(logging.INFO, logger=manifest_writer.logger.name)

    manifest_writer.dry_run_write(_records(), str(tmp_path))

    output = caplog.text
    assert "┌" in output
    assert "│ Metric    │ Count │" in output
    assert "│ Total     │ 3     │" in output
    assert "│ Train     │" in output
    assert "│ Val       │" in output
    assert "│ arxiv     │ 2     │" in output
    assert "│ wikipedia │ 1     │" in output
