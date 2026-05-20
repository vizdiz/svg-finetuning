"""
feedback.ingest

Lambda-friendly ingestion endpoint for feedback events.
The sink abstraction keeps this testable without AWS access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import boto3

from backend.feedback.gating import DownloadGateConfig, should_allow_download
from backend.feedback.schema import FeedbackIngestRequest, FeedbackIngestResponse, FeedbackRecord


class FeedbackSink(Protocol):
    def append(self, record: FeedbackRecord) -> None: ...


class LocalJsonlSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: FeedbackRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")


class S3JsonlSink:
    def __init__(self, bucket: str, key: str, region: str = "us-east-1"):
        self.bucket = bucket
        self.key = key
        self.s3 = boto3.client("s3", region_name=region)

    def append(self, record: FeedbackRecord) -> None:
        existing = b""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.key)
            existing = obj["Body"].read()
        except Exception:
            existing = b""
        body = existing + (record.to_json() + "\n").encode("utf-8")
        self.s3.put_object(Bucket=self.bucket, Key=self.key, Body=body, ContentType="application/jsonl")


def parse_feedback_event(event: dict[str, Any]) -> FeedbackRecord:
    if "body" in event:
        body = event["body"]
        payload = json.loads(body) if isinstance(body, str) else body
    else:
        payload = event
    request = FeedbackIngestRequest.from_dict(payload)
    record = FeedbackRecord(
        request_id=request.request_id,
        rating=request.rating,
        created_at=request.created_at,
        prompt=request.prompt,
        response_svg=request.response_svg,
        session_id=request.session_id,
        branch_id=request.branch_id,
        revision_index=request.revision_index,
        source=request.source,
        model_name=request.model_name,
        download_allowed=request.download_allowed,
        comment=request.comment,
        metadata=request.metadata,
    )
    record.download_allowed = should_allow_download(record, DownloadGateConfig())
    return record


def ingest_feedback(
    event: dict[str, Any],
    context: Any | None = None,
    *,
    sink: FeedbackSink | None = None,
    gate_config: DownloadGateConfig | None = None,
) -> dict[str, Any]:
    record = parse_feedback_event(event)
    if gate_config is not None:
        record.download_allowed = should_allow_download(record, gate_config)

    if sink is None:
        sink = LocalJsonlSink(Path("feedback") / "feedback.jsonl")
    sink.append(record)
    response = FeedbackIngestResponse(
        request_id=record.request_id,
        accepted=True,
        download_allowed=record.download_allowed,
        feedback_uri="",
        message="accepted",
    )
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(response.to_dict()),
    }
