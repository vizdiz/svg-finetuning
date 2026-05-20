"""
inference.router

Small request router for the generation surface.

Shape:
  API Gateway -> cache check -> endpoint client

The cache and endpoint are protocol-based so the same logic can run in a
Lambda, ECS task, or local unit test without bringing in Redis/uvicorn.
"""

from __future__ import annotations

import hashlib
import html
import json
from typing import Any, Protocol

from backend.inference.schemas import (
    ApiErrorResponse,
    GenerateRequest,
    GenerateResponse,
    UploadRequest,
    UploadResponse,
)

class CacheClient(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None: ...


class EndpointClient(Protocol):
    def generate(self, payload: dict[str, Any]) -> dict[str, Any] | str: ...


from dataclasses import dataclass


@dataclass
class RouterConfig:
    cache_ttl_seconds: int = 86_400
    endpoint_name: str = ""
    response_format: str = "svg"


RouteResult = GenerateResponse


class InMemoryCache:
    def __init__(self):
        self._data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        self._data[key] = value


def canonical_cache_key(payload: dict[str, Any]) -> str:
    allowed = {
        key: payload.get(key)
        for key in (
            "prompt",
            "model",
            "temperature",
            "style",
            "width",
            "height",
            "seed",
            "cache_namespace",
            "max_tokens",
            "session_id",
            "parent_request_id",
            "branch_id",
            "revision_index",
            "feedback",
            "feedback_rating",
        )
        if payload.get(key) is not None
    }
    raw = json.dumps(allowed, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _extract_svg(response: dict[str, Any] | str) -> tuple[str, dict[str, Any]]:
    if isinstance(response, str):
        return response, {}
    if "svg" in response and isinstance(response["svg"], str):
        meta = {k: v for k, v in response.items() if k != "svg"}
        return response["svg"], meta
    if "body" in response and isinstance(response["body"], str):
        try:
            parsed = json.loads(response["body"])
            if isinstance(parsed, dict) and isinstance(parsed.get("svg"), str):
                meta = {k: v for k, v in parsed.items() if k != "svg"}
                return parsed["svg"], meta
        except json.JSONDecodeError:
            pass
        return response["body"], {}
    raise ValueError("endpoint response does not contain SVG content")


def route_request(
    payload: dict[str, Any] | GenerateRequest,
    *,
    cache: CacheClient | None = None,
    endpoint_client: EndpointClient | None = None,
    config: RouterConfig | None = None,
) -> RouteResult:
    cache = cache or InMemoryCache()
    config = config or RouterConfig()
    endpoint_client = endpoint_client or InMemoryCacheEndpoint()

    request = payload if isinstance(payload, GenerateRequest) else GenerateRequest.from_dict(payload)
    cache_key = canonical_cache_key(request.cache_payload())
    cached_svg = cache.get(cache_key)
    if cached_svg is not None:
        return GenerateResponse(
            request_id=str(request.request_id or cache_key[:16]),
            cache_key=cache_key,
            cached=True,
            svg=cached_svg,
            model=request.model,
            metadata={
                "source": "cache",
                "cache_namespace": request.cache_namespace,
                "session_id": request.session_id,
                "branch_id": request.branch_id,
                "revision_index": request.revision_index,
                "parent_request_id": request.parent_request_id,
                "feedback": request.feedback,
                "feedback_rating": request.feedback_rating,
            },
        )

    response = endpoint_client.generate(request.to_dict())
    svg, metadata = _extract_svg(response)
    cache.set(cache_key, svg, ttl_seconds=config.cache_ttl_seconds)
    return GenerateResponse(
        request_id=str(request.request_id or cache_key[:16]),
        cache_key=cache_key,
        cached=False,
        svg=svg,
        model=request.model,
        metadata={
            **metadata,
            "cache_namespace": request.cache_namespace,
            "session_id": request.session_id,
            "branch_id": request.branch_id,
            "revision_index": request.revision_index,
            "parent_request_id": request.parent_request_id,
            "feedback": request.feedback,
            "feedback_rating": request.feedback_rating,
        },
    )


def handle_upload_request(payload: dict[str, Any] | UploadRequest) -> UploadResponse:
    request = payload if isinstance(payload, UploadRequest) else UploadRequest.from_dict(payload)
    asset_key = f"uploads/{hashlib.sha256(request.svg.encode()).hexdigest()}.svg"
    return UploadResponse(
        request_id=request.request_id or asset_key[:16],
        accepted=True,
        asset_key=asset_key,
        bytes=len(request.svg.encode()),
        message="accepted",
    )


def handle_api_gateway_event(
    event: dict[str, Any],
    context: Any | None = None,
    *,
    cache: CacheClient | None = None,
    endpoint_client: EndpointClient | None = None,
    config: RouterConfig | None = None,
) -> dict[str, Any]:
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raise ValueError("base64 encoded bodies are not supported in this stub router")
    payload = json.loads(body) if isinstance(body, str) else body
    path = str(event.get("path") or "/generate")
    try:
        if path.endswith("/upload"):
            result = handle_upload_request(payload)
        elif path.endswith("/feedback"):
            from backend.feedback.ingest import parse_feedback_event

            record = parse_feedback_event(payload)
            result = {
                "request_id": record.request_id,
                "accepted": True,
                "download_allowed": record.download_allowed,
                "feedback_uri": "",
                "message": "accepted",
            }
        else:
            result = route_request(payload, cache=cache, endpoint_client=endpoint_client, config=config)
    except Exception as exc:
        error = ApiErrorResponse(error=type(exc).__name__, message=str(exc))
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(error.to_dict()),
        }
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(result.to_dict() if hasattr(result, "to_dict") else result),
    }


class InMemoryCacheEndpoint:
    """
    Minimal local stand-in for a GPU-backed endpoint.
    It returns a predictable SVG payload so the request flow is unit-testable.
    """

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = payload.get("prompt", "")
        feedback = payload.get("feedback", "")
        revision_index = int(payload.get("revision_index", 0) or 0)
        branch_id = payload.get("branch_id", "")
        digest = hashlib.md5(f"{prompt}|{feedback}|{revision_index}|{branch_id}".encode()).hexdigest()[:8]
        prompt_line = html.escape(str(prompt)[:32])
        feedback_line = html.escape(str(feedback)[:40])
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180">'
            f'<rect x="8" y="8" width="304" height="164" fill="white" stroke="black"/>'
            f'<text x="20" y="44" font-size="16">{prompt_line}</text>'
            f'<text x="20" y="72" font-size="12">rev:{revision_index} branch:{html.escape(str(branch_id)[:12])}</text>'
            f'<text x="20" y="96" font-size="12">{feedback_line}</text>'
            f'<text x="20" y="124" font-size="12">stub:{digest}</text>'
            "</svg>"
        )
        return {"svg": svg, "request_id": payload.get("request_id", digest)}
