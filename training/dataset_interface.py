"""
dataset_interface.py
Contract between the dataset generation pipeline and the training pipeline.

The dataset pipeline (not yet implemented) is responsible for:
  1. Generating/aggregating SVG training examples
  2. Writing them as JSONL files under s3://<data-bucket>/train/<batch-id>/
  3. Writing a DatasetManifest to s3://<data-bucket>/train/dataset_manifest.json

Writing the manifest is the signal that triggers training. The training
pipeline reads nothing else from the dataset pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import json
from datetime import datetime, timezone


@dataclass
class DatasetManifest:
    """
    Written by the dataset pipeline to signal a batch is ready.
    Read by the training pipeline to locate and validate data.
    """
    dataset_id: str                 # unique identifier for this batch
    created_at: str                 # ISO-8601 UTC timestamp
    record_count: int               # total number of training examples
    files: List[str]                # s3:// URIs of JSONL files, in order
    schema_version: str = "1.0"
    split: str = "train"
    # Optional metadata — dataset pipeline can add anything here;
    # training pipeline ignores unknown keys
    metadata: dict = field(default_factory=dict)

    # ── Serialization ──────────────────────────────────────────────────────
    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "DatasetManifest":
        d = json.loads(raw)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    # ── Validation ─────────────────────────────────────────────────────────
    def validate(self) -> None:
        if not self.files:
            raise ValueError("Manifest has no files listed")
        if self.record_count <= 0:
            raise ValueError(f"record_count must be > 0, got {self.record_count}")
        for uri in self.files:
            if not uri.startswith("s3://"):
                raise ValueError(f"Expected s3:// URI, got: {uri}")


@dataclass
class TrainingRecord:
    """
    Schema for a single record in a JSONL training file.
    Each line in every file listed in the manifest must deserialize to this.
    """
    prompt: str     # natural language description of the SVG to generate
    svg: str        # the target SVG string (well-formed XML)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingRecord":
        missing = [f for f in ("prompt", "svg") if f not in d]
        if missing:
            raise ValueError(f"Training record missing fields: {missing}")
        return cls(prompt=d["prompt"], svg=d["svg"])


class DatasetLoader:
    """
    Loads a dataset described by a DatasetManifest.
    The dataset pipeline never interacts with this class — it only writes
    the manifest and JSONL files. Swap the implementation here when the
    data format or storage layer changes without touching train.py.
    """

    def __init__(self, manifest: DatasetManifest, s3_client=None):
        import boto3
        self.manifest = manifest
        self.s3 = s3_client or boto3.client("s3")

    def iter_records(self):
        """Yields TrainingRecord one at a time, streaming from S3."""
        for uri in self.manifest.files:
            bucket, key = uri[len("s3://"):].split("/", 1)
            resp = self.s3.get_object(Bucket=bucket, Key=key)
            for line in resp["Body"].iter_lines():
                if line:
                    yield TrainingRecord.from_dict(json.loads(line))

    def as_hf_dataset(self, tokenizer, max_length: int = 1024):
        """
        Returns a HuggingFace Dataset ready for Trainer.
        Format: <prompt>\n\n<svg>
        """
        from datasets import Dataset

        records = [
            {"text": f"{r.prompt}\n\n{r.svg}"}
            for r in self.iter_records()
        ]
        if not records:
            raise ValueError(f"Dataset '{self.manifest.dataset_id}' loaded 0 records")

        dataset = Dataset.from_list(records)

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )

        return dataset.map(tokenize, batched=True, remove_columns=["text"])


# ── S3 manifest helpers ───────────────────────────────────────────────────────

MANIFEST_KEY = "train/dataset_manifest.json"

def read_manifest_from_s3(bucket: str, s3_client=None) -> DatasetManifest:
    import boto3
    s3 = s3_client or boto3.client("s3")
    raw = s3.get_object(Bucket=bucket, Key=MANIFEST_KEY)["Body"].read().decode()
    manifest = DatasetManifest.from_json(raw)
    manifest.validate()
    return manifest


def write_manifest_to_s3(manifest: DatasetManifest, bucket: str, s3_client=None) -> None:
    """Called by the dataset pipeline — not by training."""
    import boto3
    s3 = s3_client or boto3.client("s3")
    manifest.validate()
    s3.put_object(
        Bucket=bucket,
        Key=MANIFEST_KEY,
        Body=manifest.to_json().encode(),
        ContentType="application/json",
    )
