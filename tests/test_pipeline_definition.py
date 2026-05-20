from __future__ import annotations

import json
from types import SimpleNamespace

from backend.training import pipeline_definition
from backend.training.model_registry import ModelPackageGroupSpec, ensure_model_package_group


def test_default_pipeline_definition_has_expected_order():
    spec = pipeline_definition.default_pipeline_definition(
        project_name="svg-finetuning",
        role_arn="arn:aws:iam::123:role/test",
        script_s3_prefix="s3://bucket/scripts/training/",
        model_package_group_name="svg-finetuning-models",
        training_image="train-image",
        evaluation_image="eval-image",
        data_bucket="data",
        models_bucket="models",
    )

    order = pipeline_definition.validate_pipeline_dag(spec)
    assert order == ["Preprocess", "Train", "Evaluate", "RegisterModel"]


def test_upsert_pipeline_writes_json(tmp_path):
    spec = pipeline_definition.default_pipeline_definition(
        project_name="svg-finetuning",
        role_arn="arn:aws:iam::123:role/test",
        script_s3_prefix="s3://bucket/scripts/training/",
        model_package_group_name="svg-finetuning-models",
        training_image="train-image",
        evaluation_image="eval-image",
        data_bucket="data",
        models_bucket="models",
    )

    out = pipeline_definition.upsert_pipeline(spec, output_path=tmp_path / "pipeline.json")
    payload = json.loads((tmp_path / "pipeline.json").read_text())

    assert out["pipeline_name"] == "svg-finetuning-training-pipeline"
    assert payload["steps"][0]["name"] == "Preprocess"


def test_model_package_group_helper_uses_client():
    calls = {}

    class _FakeClient:
        def create_model_package_group(self, **kwargs):
            calls["kwargs"] = kwargs
            return {"ModelPackageGroupArn": "arn:aws:sagemaker:::model-package-group/test"}

    spec = ModelPackageGroupSpec(
        name="svg-finetuning-models",
        description="SVG finetuning models",
        tags={"Project": "svg-finetuning"},
    )
    result = ensure_model_package_group(spec, client=_FakeClient())

    assert result["model_package_group_name"] == "svg-finetuning-models"
    assert calls["kwargs"]["ModelPackageGroupName"] == "svg-finetuning-models"
    assert calls["kwargs"]["Tags"] == [{"Key": "Project", "Value": "svg-finetuning"}]

