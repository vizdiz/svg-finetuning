"""
training.model_registry

Helpers for creating and validating the SageMaker Model Package Group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import boto3


@dataclass
class ModelPackageGroupSpec:
    name: str
    description: str
    tags: dict[str, str]


def ensure_model_package_group(
    spec: ModelPackageGroupSpec,
    *,
    region: str = "us-east-1",
    client: Any | None = None,
) -> dict[str, Any]:
    sm = client or boto3.client("sagemaker", region_name=region)
    try:
        response = sm.create_model_package_group(
            ModelPackageGroupName=spec.name,
            ModelPackageGroupDescription=spec.description,
            Tags=[{"Key": key, "Value": value} for key, value in spec.tags.items()],
        )
    except Exception as exc:
        if "already exists" not in str(exc).lower() and "resourceinuse" not in str(exc).lower():
            raise
        response = {}
    return {
        "model_package_group_name": spec.name,
        "description": spec.description,
        "tags": spec.tags,
        "response": response,
    }

