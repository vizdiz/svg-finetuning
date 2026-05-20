"""
feedback.gating

Simple rate-before-download logic.

This is intentionally deterministic and conservative: a user must submit a
rating before download is unlocked for the request they just reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.feedback.schema import FeedbackRecord


@dataclass
class DownloadGateConfig:
    minimum_rating: int = 3
    require_feedback: bool = True


def should_allow_download(
    feedback: FeedbackRecord | None,
    config: DownloadGateConfig | None = None,
) -> bool:
    config = config or DownloadGateConfig()
    if feedback is None:
        return not config.require_feedback
    return int(feedback.rating) >= int(config.minimum_rating)


def gate_payload(payload: dict, config: DownloadGateConfig | None = None) -> dict:
    feedback = None
    if payload.get("feedback"):
        feedback = FeedbackRecord.from_dict(payload["feedback"])
    payload = dict(payload)
    payload["download_allowed"] = should_allow_download(feedback, config)
    return payload

