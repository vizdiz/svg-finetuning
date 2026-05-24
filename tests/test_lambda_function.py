"""
Unit tests for lambda/lambda_function.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx


def _load_lambda_module(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    path = Path(__file__).resolve().parents[1] / "lambda" / "lambda_function.py"
    spec = importlib.util.spec_from_file_location("lambda_function_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_lambda_uses_env_driven_resource_names(monkeypatch):
    module = _load_lambda_module(
        monkeypatch,
        ACCOUNT_ID="123456789012",
        AWS_REGION="us-east-1",
        DATA_BUCKET="data-bucket",
        MODELS_BUCKET="models-bucket",
        SM_ROLE="arn:aws:iam::123456789012:role/test",
        SCRIPTS_URI="s3://scripts-bucket/training/",
        ENDPOINT_NAME="endpoint",
    )

    assert module.DATA_BUCKET == "data-bucket"
    assert module.MODELS_BUCKET == "models-bucket"
    assert module.SM_ROLE == "arn:aws:iam::123456789012:role/test"
    assert module.SCRIPTS_URI == "s3://scripts-bucket/training/"


def test_validate_manifest_accepts_expected_schema(monkeypatch):
    module = _load_lambda_module(
        monkeypatch,
        ACCOUNT_ID="123456789012",
        AWS_REGION="us-east-1",
        DATA_BUCKET="data-bucket",
        MODELS_BUCKET="models-bucket",
        SM_ROLE="arn:aws:iam::123456789012:role/test",
        SCRIPTS_URI="s3://scripts-bucket/training/",
        ENDPOINT_NAME="endpoint",
    )

    class _FakeBody:
        def read(self):
            return b'{"dataset_id":"x","created_at":"2025-01-01T00:00:00Z","record_count":1,"files":["s3://a"],"schema_version":"1.0","split":"train"}'

    class _FakeS3:
        def get_object(self, **kwargs):
            assert kwargs["Bucket"] == "data-bucket"
            assert kwargs["Key"] == "train/dataset_manifest.json"
            return {"Body": _FakeBody()}

    monkeypatch.setattr(module.boto3, "client", lambda service, region_name=None: _FakeS3())

    manifest = module.validate_manifest("data-bucket")
    assert manifest["dataset_id"] == "x"
    assert manifest["record_count"] == 1


def test_validate_manifest_rejects_bad_manifest(monkeypatch):
    module = _load_lambda_module(
        monkeypatch,
        ACCOUNT_ID="123456789012",
        AWS_REGION="us-east-1",
        DATA_BUCKET="data-bucket",
        MODELS_BUCKET="models-bucket",
        SM_ROLE="arn:aws:iam::123456789012:role/test",
        SCRIPTS_URI="s3://scripts-bucket/training/",
        ENDPOINT_NAME="endpoint",
    )

    class _FakeBody:
        def read(self):
            return b'{"dataset_id":"x","record_count":0,"files":[]}'

    class _FakeS3:
        def get_object(self, **kwargs):
            return {"Body": _FakeBody()}

    monkeypatch.setattr(module.boto3, "client", lambda service, region_name=None: _FakeS3())

    try:
        module.validate_manifest("data-bucket")
    except ValueError as exc:
        assert "Manifest lists no data files" in str(exc) or "record_count" in str(exc)
    else:
        raise AssertionError("validate_manifest should reject malformed manifests")


def test_handler_accepts_s3_event_and_starts_training(monkeypatch):
    module = _load_lambda_module(
        monkeypatch,
        ACCOUNT_ID="123456789012",
        AWS_REGION="us-east-1",
        DATA_BUCKET="data-bucket",
        MODELS_BUCKET="models-bucket",
        SM_ROLE="arn:aws:iam::123456789012:role/test",
        SCRIPTS_URI="s3://scripts-bucket/training/",
        ENDPOINT_NAME="endpoint",
        HF_SECRET_ID="secret",
        TRAINING_IMAGE="image",
    )

    class _FakeBody:
        def read(self):
            return b'{"dataset_id":"x","created_at":"2025-01-01T00:00:00Z","record_count":1,"files":["s3://a"],"schema_version":"1.0","split":"train"}'

    class _FakeS3:
        def get_object(self, **kwargs):
            return {"Body": _FakeBody()}

    class _FakeSecrets:
        def get_secret_value(self, **kwargs):
            return {"SecretString": "token"}

    class _FakeSM:
        def create_training_job(self, **kwargs):
            self.kwargs = kwargs
            return {"TrainingJobArn": "arn:aws:sagemaker:job/test"}

    fake_sm = _FakeSM()

    def fake_client(service, region_name=None):
        if service == "s3":
            return _FakeS3()
        if service == "secretsmanager":
            return _FakeSecrets()
        if service == "sagemaker":
            return fake_sm
        raise AssertionError(service)

    monkeypatch.setattr(module.boto3, "client", fake_client)

    response = module.handler({"Records": [{"eventSource": "aws:s3"}]}, None)

    assert response["statusCode"] == 200
    assert fake_sm.kwargs["TrainingJobName"].startswith("svg-finetune-")
    assert fake_sm.kwargs["HyperParameters"]["num_train_epochs"] == "1"
    assert fake_sm.kwargs["HyperParameters"]["per_device_train_batch_size"] == "1"
    assert fake_sm.kwargs["HyperParameters"]["max_length"] == "4096"
    assert fake_sm.kwargs["HyperParameters"]["max_target_chars"] == "100000"
    assert fake_sm.kwargs["HyperParameters"]["drop_overlength_records"] == "true"
    assert fake_sm.kwargs["HyperParameters"]["gradient_checkpointing"] == "true"
    assert fake_sm.kwargs["HyperParameters"]["update_endpoint"] == "false"
    assert fake_sm.kwargs["ResourceConfig"]["InstanceType"] == "ml.g5.2xlarge"
    assert fake_sm.kwargs["Environment"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
