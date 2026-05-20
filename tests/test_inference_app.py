from __future__ import annotations

import json

from backend.inference.app import handle_api_gateway_event, handle_feedback_api, handle_generate_api
from backend.inference.router import InMemoryCache, InMemoryCacheEndpoint
from backend.inference.sessions import InMemorySessionStore


def test_generate_api_persists_session_and_branch_context():
    cache = InMemoryCache()
    endpoint = InMemoryCacheEndpoint()
    store = InMemorySessionStore()

    result = handle_generate_api(
        {
            "request_id": "req-1",
            "session_id": "sess-1",
            "branch_id": "branch-a",
            "revision_index": 0,
            "prompt": "draw a diagram",
            "model": "qwen",
        },
        cache=cache,
        endpoint_client=endpoint,
        session_store=store,
    )

    session = store.load("sess-1")
    assert result.metadata["session_id"] == "sess-1"
    assert session is not None
    assert session.current_branch_id == "branch-a"
    assert session.generations[0].request_id == "req-1"


def test_generate_api_normalizes_media_inputs_before_generation():
    cache = InMemoryCache()
    endpoint = InMemoryCacheEndpoint()
    store = InMemorySessionStore()

    result = handle_generate_api(
        {
            "request_id": "req-media-1",
            "session_id": "sess-media-1",
            "branch_id": "branch-media",
            "revision_index": 0,
            "prompt": "",
            "input_mode": "image",
            "reference_images": ["s3://bucket/reference.png"],
            "media_metadata": {
                "s3://bucket/reference.png": {
                    "filename": "reference.png",
                    "mime_type": "image/png",
                    "width": 1200,
                    "height": 800,
                    "caption": "reference diagram",
                }
            },
        },
        cache=cache,
        endpoint_client=endpoint,
        session_store=store,
    )

    session = store.load("sess-media-1")
    assert result.metadata["input_mode"] == "image"
    assert "diagram_description" in result.metadata
    assert session is not None
    assert session.prompt == ""
    assert session.generations[0].prompt == ""


def test_feedback_api_updates_session_history():
    store = InMemorySessionStore()
    handle_generate_api(
        {
            "request_id": "req-1",
            "session_id": "sess-1",
            "branch_id": "branch-a",
            "revision_index": 0,
            "prompt": "draw a diagram",
            "model": "qwen",
        },
        cache=InMemoryCache(),
        endpoint_client=InMemoryCacheEndpoint(),
        session_store=store,
    )

    response = handle_feedback_api(
        {
            "request_id": "fb-1",
            "rating": 4,
            "prompt": "draw a diagram",
            "response_svg": "<svg/>",
            "created_at": "2026-01-01T00:00:00Z",
            "comment": "good",
            "metadata": {"session_id": "sess-1"},
        },
        session_store=store,
    )

    session = store.load("sess-1")
    assert response["download_allowed"] is True
    assert session is not None
    assert session.feedback[0].request_id == "fb-1"


def test_api_gateway_event_routes_generate_and_feedback():
    store = InMemorySessionStore()
    generate_event = {
        "path": "/api/generate",
        "body": json.dumps({"request_id": "req-1", "session_id": "sess-1", "prompt": "draw a diagram"}),
    }
    generate_response = handle_api_gateway_event(
        generate_event,
        session_store=store,
        cache=InMemoryCache(),
        endpoint_client=InMemoryCacheEndpoint(),
    )
    feedback_event = {
        "path": "/api/feedback",
        "body": json.dumps(
            {
                "request_id": "fb-1",
                "rating": 4,
                "prompt": "draw a diagram",
                "response_svg": "<svg/>",
                "metadata": {"session_id": "sess-1"},
            }
        ),
    }
    feedback_response = handle_api_gateway_event(feedback_event, session_store=store)

    assert generate_response["statusCode"] == 200
    assert feedback_response["statusCode"] == 200
