"""
backend.inference.app

Service-layer handlers for the cache-first generation API.
This is the actual backend wiring used by Lambda and local tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import uuid

from backend.feedback.ingest import parse_feedback_event
from backend.inference.contracts import ModelGenerationRequest
from backend.inference.endpoint_clients import SageMakerVLLMEndpointClient
from backend.inference.router import (
    CacheClient,
    EndpointClient,
    InMemoryCache,
    InMemoryCacheEndpoint,
    RouterConfig,
    handle_upload_request,
    route_request,
)
from backend.inference.schemas import ApiErrorResponse, GenerateRequest, GenerateResponse, UploadResponse
from backend.inference.sessions import (
    DynamoDBSessionStore,
    InMemorySessionStore,
    LocalJsonSessionStore,
    SessionFeedback,
    SessionGeneration,
    SessionRecord,
    SessionStore,
)


@dataclass
class InferenceAppConfig:
    cache_ttl_seconds: int = 86_400
    session_store_mode: str = "memory"
    session_store_path: str = "sessions"
    session_table_name: str = ""
    endpoint_mode: str = "mock"
    endpoint_model: str = "Qwen/Qwen2.5-7B-Instruct"
    endpoint_name: str = ""
    endpoint_timeout_s: float = 60.0
    region: str = "us-east-1"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _build_config() -> InferenceAppConfig:
    return InferenceAppConfig(
        cache_ttl_seconds=_env_int("CACHE_TTL_SECONDS", 86400),
        session_store_mode=os.environ.get("SESSION_STORE_MODE", "memory"),
        session_store_path=os.environ.get("SESSION_STORE_PATH", "sessions"),
        session_table_name=os.environ.get("SESSION_TABLE_NAME", ""),
        endpoint_mode=os.environ.get("ENDPOINT_MODE", "mock"),
        endpoint_model=os.environ.get("ENDPOINT_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        endpoint_name=os.environ.get("ENDPOINT_NAME", ""),
        endpoint_timeout_s=float(os.environ.get("ENDPOINT_TIMEOUT_S", "60")),
        region=os.environ.get("AWS_REGION", os.environ.get("REGION", "us-east-1")),
    )


def build_cache() -> CacheClient:
    return InMemoryCache()


def build_endpoint_client(config: InferenceAppConfig):
    if config.endpoint_mode == "sagemaker":
        if not config.endpoint_name:
            raise RuntimeError("ENDPOINT_NAME is required when ENDPOINT_MODE=sagemaker")
        return SageMakerVLLMEndpointClient(
            endpoint_name=config.endpoint_name,
            region=config.region,
            timeout_s=config.endpoint_timeout_s,
        )
    return InMemoryCacheEndpoint()


def build_session_store(config: InferenceAppConfig) -> SessionStore:
    if config.session_store_mode == "dynamodb" and config.session_table_name:
        return DynamoDBSessionStore(config.session_table_name, region=config.region)
    if config.session_store_mode == "local":
        return LocalJsonSessionStore(Path(config.session_store_path))
    return InMemorySessionStore()


def _session_from_request(
    request: GenerateRequest,
    response: GenerateResponse,
    *,
    prompt: str,
) -> SessionGeneration:
    metadata = dict(response.metadata)
    return SessionGeneration(
        request_id=response.request_id,
        cache_key=response.cache_key,
        branch_id=request.branch_id or metadata.get("branch_id", ""),
        revision_index=int(request.revision_index),
        parent_request_id=request.parent_request_id,
        prompt=prompt,
        feedback=request.feedback,
        feedback_rating=request.feedback_rating,
        model=response.model or request.model,
        cached=response.cached,
        svg=response.svg,
        metadata=metadata,
    )


def _load_or_create_session(store: SessionStore, session_id: str, prompt: str) -> SessionRecord:
    if session_id:
        existing = store.load(session_id)
        if existing is not None:
            if not existing.prompt:
                existing.prompt = prompt
            return existing
    new_session_id = session_id or uuid.uuid4().hex
    return SessionRecord(session_id=new_session_id, prompt=prompt)


def handle_generate_api(
    payload: dict[str, Any] | GenerateRequest,
    *,
    cache: CacheClient | None = None,
    endpoint_client: EndpointClient | None = None,
    session_store: SessionStore | None = None,
    config: InferenceAppConfig | None = None,
) -> GenerateResponse:
    config = config or _build_config()
    cache = cache or build_cache()
    endpoint_client = endpoint_client or build_endpoint_client(config)
    session_store = session_store or build_session_store(config)

    request = payload if isinstance(payload, GenerateRequest) else GenerateRequest.from_dict(payload)
    if not request.session_id:
        request.session_id = uuid.uuid4().hex
    if not request.branch_id:
        request.branch_id = uuid.uuid4().hex[:12]

    response = route_request(
        request,
        cache=cache,
        endpoint_client=endpoint_client,
        config=RouterConfig(cache_ttl_seconds=config.cache_ttl_seconds),
    )
    session = _load_or_create_session(session_store, request.session_id, request.prompt)
    session.append_generation(_session_from_request(request, response, prompt=request.prompt))
    session_store.save(session)
    response.metadata = {
        **response.metadata,
        "session_id": session.session_id,
        "branch_id": request.branch_id,
        "revision_index": request.revision_index,
        "parent_request_id": request.parent_request_id,
        "session_updated_at": session.updated_at,
    }
    response.metadata.setdefault("model_request", ModelGenerationRequest.from_dict(request.to_dict()).to_dict())
    return response


def handle_feedback_api(
    payload: dict[str, Any],
    *,
    session_store: SessionStore | None = None,
    config: InferenceAppConfig | None = None,
) -> dict[str, Any]:
    config = config or _build_config()
    session_store = session_store or build_session_store(config)
    record = parse_feedback_event(payload)

    session_id = str(record.session_id or record.metadata.get("session_id", ""))
    if session_id:
        session = _load_or_create_session(session_store, session_id, record.prompt)
        session.append_feedback(
            SessionFeedback(
                request_id=record.request_id,
                rating=record.rating,
                prompt=record.prompt,
                response_svg=record.response_svg,
                created_at=record.created_at,
                source=record.source,
                model_name=record.model_name,
                comment=record.comment,
                download_allowed=record.download_allowed,
                metadata=record.metadata,
            )
        )
        session_store.save(session)

    response = {
        "request_id": record.request_id,
        "accepted": True,
        "download_allowed": record.download_allowed,
        "feedback_uri": "",
        "message": "accepted",
        "session_id": session_id,
        "metadata": record.metadata,
    }
    return response


def handle_upload_api(payload: dict[str, Any]) -> UploadResponse:
    request = handle_upload_request(payload)
    return request


def handle_api_gateway_event(
    event: dict[str, Any],
    context: Any | None = None,
    *,
    cache: CacheClient | None = None,
    endpoint_client: EndpointClient | None = None,
    session_store: SessionStore | None = None,
    config: InferenceAppConfig | None = None,
) -> dict[str, Any]:
    config = config or _build_config()
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raise ValueError("base64 encoded bodies are not supported in this stub router")
    payload = json.loads(body) if isinstance(body, str) else body
    path = str(event.get("path") or "/api/generate")
    try:
        if path.endswith("/upload"):
            result = handle_upload_api(payload)
        elif path.endswith("/feedback"):
            result = handle_feedback_api(payload, session_store=session_store, config=config)
        else:
            result = handle_generate_api(
                payload,
                cache=cache,
                endpoint_client=endpoint_client,
                session_store=session_store,
                config=config,
            )
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
