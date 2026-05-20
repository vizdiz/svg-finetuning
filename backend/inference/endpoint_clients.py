"""
backend.inference.endpoint_clients

SageMaker runtime adapters for the internal vLLM contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

import boto3

from backend.inference.contracts import ModelGenerationRequest


@dataclass
class SageMakerVLLMEndpointClient:
    endpoint_name: str
    region: str = "us-east-1"
    timeout_s: float = 60.0
    runtime: Any | None = None

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = ModelGenerationRequest.from_dict(payload)
        runtime = self.runtime or boto3.client("sagemaker-runtime", region_name=self.region)
        response = runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(request.to_dict()).encode(),
        )
        body = response["Body"].read()
        parsed: dict[str, Any]
        try:
            parsed = json.loads(body.decode())
        except Exception:
            parsed = {"svg": body.decode()}

        svg = parsed.get("svg")
        if not isinstance(svg, str) or not svg.strip():
            raise ValueError("endpoint response did not include SVG content")

        return {
            "svg": svg,
            "request_id": parsed.get("request_id", request.request_id),
            "model": parsed.get("model", request.model),
            "metadata": parsed.get("metadata", {}),
        }


@dataclass
class SageMakerMediaDescriptionEndpointClient:
    """
    Adapter for a vision-enabled SageMaker endpoint that converts uploaded
    media into a structured diagram description.

    The endpoint contract is intentionally lightweight:
      - accept the repo's JSON request shape
      - return JSON with diagram_description and optional metadata
    """

    endpoint_name: str
    region: str = "us-east-1"
    timeout_s: float = 60.0
    runtime: Any | None = None

    def describe(self, payload: dict[str, Any]) -> dict[str, Any]:
        runtime = self.runtime or boto3.client("sagemaker-runtime", region_name=self.region)
        response = runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload).encode(),
        )
        body = response["Body"].read()
        try:
            parsed = json.loads(body.decode())
        except Exception:
            parsed = {
                "diagram_description": body.decode(),
            }

        if not isinstance(parsed, dict):
            raise ValueError("vision endpoint response must be JSON")

        diagram_description = str(
            parsed.get("diagram_description")
            or parsed.get("description")
            or parsed.get("caption")
            or ""
        ).strip()
        if not diagram_description:
            raise ValueError("vision endpoint response did not include a diagram description")

        media_summary = parsed.get("media_summary")
        if media_summary is None:
            media_summary = []
        if not isinstance(media_summary, list):
            media_summary = [str(media_summary)]

        metadata = parsed.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        return {
            "diagram_description": diagram_description,
            "media_summary": media_summary,
            "metadata": metadata,
        }
