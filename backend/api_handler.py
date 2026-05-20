"""
Lambda entrypoint for the public API surface.

The Lambda package is intentionally thin: the real request handling lives
in backend.inference.app so the same code can run locally, in Lambda, or
behind API Gateway + CloudFront.
"""

from __future__ import annotations

from backend.inference.app import handle_api_gateway_event


def handler(event, context):
    return handle_api_gateway_event(event, context)
