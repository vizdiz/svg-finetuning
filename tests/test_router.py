from __future__ import annotations

import json

from backend.inference.router import InMemoryCache, InMemoryCacheEndpoint, handle_api_gateway_event, route_request
from backend.inference.schemas import GenerateRequest, UploadRequest


def test_route_request_hits_cache_on_second_call():
    cache = InMemoryCache()
    endpoint = InMemoryCacheEndpoint()
    payload = GenerateRequest(request_id="abc", prompt="draw a diagram", model="qwen")

    first = route_request(payload, cache=cache, endpoint_client=endpoint)
    second = route_request(payload, cache=cache, endpoint_client=endpoint)

    assert first.cached is False
    assert second.cached is True
    assert first.svg == second.svg


def test_route_request_revision_context_uses_separate_cache_entry():
    cache = InMemoryCache()
    endpoint = InMemoryCacheEndpoint()
    base = GenerateRequest(request_id="abc", prompt="draw a diagram", model="qwen")
    revised = GenerateRequest(
        request_id="abc-2",
        prompt="draw a diagram",
        model="qwen",
        parent_request_id="abc",
        branch_id="branch-a",
        revision_index=1,
        feedback="route connectors more cleanly",
    )

    first = route_request(base, cache=cache, endpoint_client=endpoint)
    second = route_request(revised, cache=cache, endpoint_client=endpoint)
    third = route_request(revised, cache=cache, endpoint_client=endpoint)

    assert first.cached is False
    assert second.cached is False
    assert third.cached is True
    assert first.svg != second.svg


def test_handle_api_gateway_event_returns_json_body():
    event = {"path": "/generate", "body": json.dumps({"request_id": "abc", "prompt": "technical block diagram"})}
    response = handle_api_gateway_event(event)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["request_id"] == "abc"
    assert body["svg"].startswith("<svg")


def test_handle_api_gateway_event_upload_route_returns_upload_schema():
    event = {
        "path": "/upload",
        "body": json.dumps({"request_id": "u1", "filename": "diagram.svg", "svg": "<svg/>"}),
    }
    response = handle_api_gateway_event(event)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["accepted"] is True
    assert body["asset_key"].endswith(".svg")


def test_generate_request_validation_rejects_empty_prompt():
    try:
        GenerateRequest.from_dict({"prompt": "  "})
    except ValueError as exc:
        assert "prompt or media input is required" in str(exc)
    else:
        raise AssertionError("empty prompt should be rejected")


def test_upload_request_validation_rejects_missing_svg():
    try:
        UploadRequest.from_dict({"filename": "diagram.svg"})
    except ValueError as exc:
        assert "svg is required" in str(exc)
    else:
        raise AssertionError("missing svg should be rejected")
