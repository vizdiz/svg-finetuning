from __future__ import annotations

import json

from backend.feedback.gating import DownloadGateConfig, gate_payload, should_allow_download
from backend.feedback.ingest import ingest_feedback, parse_feedback_event
from backend.feedback.schema import FeedbackIngestRequest, FeedbackRecord


def test_feedback_record_validation_and_json_round_trip():
    record = FeedbackRecord(
        request_id="req-1",
        rating=5,
        created_at="2026-01-01T00:00:00Z",
        prompt="caption",
        response_svg="<svg/>",
    )
    payload = json.loads(record.to_json())

    assert payload["rating"] == 5
    assert FeedbackRecord.from_dict(payload).request_id == "req-1"


def test_download_gate_requires_rating():
    record = FeedbackRecord(
        request_id="req-1",
        rating=2,
        created_at="2026-01-01T00:00:00Z",
        prompt="caption",
        response_svg="<svg/>",
    )
    assert not should_allow_download(record, DownloadGateConfig(minimum_rating=3))
    assert should_allow_download(record, DownloadGateConfig(minimum_rating=2))


def test_ingest_feedback_appends_to_sink(tmp_path):
    written = []

    class _Sink:
        def append(self, record):
            written.append(record)

    event = {
        "body": json.dumps(
            {
                "request_id": "req-1",
                "rating": 4,
                "created_at": "2026-01-01T00:00:00Z",
                "prompt": "caption",
                "response_svg": "<svg/>",
                "session_id": "sess-1",
                "branch_id": "branch-1",
                "revision_index": 2,
            }
        )
    }
    response = ingest_feedback(event, sink=_Sink())

    assert response["statusCode"] == 200
    assert written[0].download_allowed is True
    assert written[0].session_id == "sess-1"


def test_gate_payload_marks_download_allowed():
    payload = gate_payload(
        {
            "feedback": {
                "request_id": "req-1",
                "rating": 5,
                "created_at": "2026-01-01T00:00:00Z",
                "prompt": "caption",
                "response_svg": "<svg/>",
            }
        }
    )

    assert payload["download_allowed"] is True


def test_feedback_ingest_request_validation():
    request = FeedbackIngestRequest.from_dict(
        {
            "request_id": "req-1",
            "rating": 4,
            "prompt": "caption",
            "response_svg": "<svg/>",
            "session_id": "sess-1",
            "branch_id": "branch-1",
        }
    )

    assert request.download_allowed is False
    assert request.request_id == "req-1"
