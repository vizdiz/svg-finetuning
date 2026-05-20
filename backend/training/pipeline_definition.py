"""
training.pipeline_definition

Local SageMaker pipeline definition helpers.

This module keeps the pipeline DAG explicit and testable without requiring
the SageMaker SDK at import time. The emitted structure is intentionally
simple: the repo can validate the DAG, persist the definition, and later
adapt it to SageMaker Pipelines or another orchestrator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass
class PipelineStepSpec:
    name: str
    step_type: str
    script_uri: str
    depends_on: list[str] = field(default_factory=list)
    arguments: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineDefinitionSpec:
    pipeline_name: str
    role_arn: str
    model_package_group_name: str
    steps: list[PipelineStepSpec]
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "role_arn": self.role_arn,
            "model_package_group_name": self.model_package_group_name,
            "steps": [asdict(step) for step in self.steps],
            "parameters": self.parameters,
            "tags": self.tags,
        }


def default_pipeline_definition(
    *,
    project_name: str,
    role_arn: str,
    script_s3_prefix: str,
    model_package_group_name: str,
    training_image: str,
    evaluation_image: str,
    data_bucket: str,
    models_bucket: str,
) -> PipelineDefinitionSpec:
    preprocess = PipelineStepSpec(
        name="Preprocess",
        step_type="Processing",
        script_uri=f"{script_s3_prefix.rstrip('/')}/preprocess.py",
        arguments={
            "input_manifest": f"s3://{data_bucket}/train/dataset_manifest.json",
            "output_prefix": f"s3://{models_bucket}/pipeline-artifacts/preprocess/",
        },
    )
    train = PipelineStepSpec(
        name="Train",
        step_type="Training",
        script_uri=f"{script_s3_prefix.rstrip('/')}/train.py",
        depends_on=["Preprocess"],
        arguments={
            "training_image": training_image,
            "input_data": f"s3://{data_bucket}/train/",
            "model_output": f"s3://{models_bucket}/pipeline-artifacts/train/",
        },
    )
    evaluate = PipelineStepSpec(
        name="Evaluate",
        step_type="Processing",
        script_uri=f"{script_s3_prefix.rstrip('/')}/evaluate.py",
        depends_on=["Train"],
        arguments={
            "model_artifacts": f"s3://{models_bucket}/pipeline-artifacts/train/model.tar.gz",
            "input_data": f"s3://{data_bucket}/val/",
            "output_prefix": f"s3://{models_bucket}/pipeline-artifacts/evaluate/",
        },
        environment={
            "EVALUATION_IMAGE": evaluation_image,
        },
    )
    register = PipelineStepSpec(
        name="RegisterModel",
        step_type="ModelRegistration",
        script_uri="",
        depends_on=["Evaluate"],
        arguments={
            "model_package_group_name": model_package_group_name,
            "model_artifacts": f"s3://{models_bucket}/pipeline-artifacts/train/model.tar.gz",
        },
    )
    return PipelineDefinitionSpec(
        pipeline_name=f"{project_name}-training-pipeline",
        role_arn=role_arn,
        model_package_group_name=model_package_group_name,
        steps=[preprocess, train, evaluate, register],
        parameters={
            "project_name": project_name,
            "data_bucket": data_bucket,
            "models_bucket": models_bucket,
        },
        tags={
            "Project": project_name,
            "ManagedBy": "terraform",
        },
    )


def validate_pipeline_dag(definition: PipelineDefinitionSpec | dict[str, Any]) -> list[str]:
    spec = definition.to_dict() if isinstance(definition, PipelineDefinitionSpec) else definition
    steps = spec.get("steps", [])
    names = [step["name"] for step in steps]
    if len(names) != len(set(names)):
        raise ValueError("Pipeline step names must be unique")
    missing = []
    for step in steps:
        for dep in step.get("depends_on", []):
            if dep not in names:
                missing.append(f"{step['name']} -> {dep}")
    if missing:
        raise ValueError(f"Pipeline has unknown dependencies: {missing}")

    graph = {step["name"]: set(step.get("depends_on", [])) for step in steps}
    visiting: set[str] = set()
    visited: set[str] = set()
    order: list[str] = []

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ValueError(f"Pipeline has a dependency cycle at {node}")
        visiting.add(node)
        for dep in graph[node]:
            visit(dep)
        visiting.remove(node)
        visited.add(node)
        order.append(node)

    for name in names:
        visit(name)
    return order


def write_pipeline_definition(
    definition: PipelineDefinitionSpec | dict[str, Any],
    output_path: str | Path,
) -> Path:
    spec = definition.to_dict() if isinstance(definition, PipelineDefinitionSpec) else definition
    validate_pipeline_dag(spec)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(spec, indent=2, sort_keys=True))
    return output_path


def upsert_pipeline(
    definition: PipelineDefinitionSpec | dict[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Validate the DAG and optionally persist the definition.

    This intentionally does not start a pipeline execution. It is the
    lightweight "upsert" path this repo needs while the AWS pieces are
    still being staged.
    """
    spec = definition.to_dict() if isinstance(definition, PipelineDefinitionSpec) else definition
    validate_pipeline_dag(spec)
    if output_path is not None:
        write_pipeline_definition(spec, output_path)
    return spec

