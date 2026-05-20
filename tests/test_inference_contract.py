from __future__ import annotations

import json

from backend.inference.contracts import ModelGenerationRequest
from backend.inference.endpoint_clients import (
    SageMakerMediaDescriptionEndpointClient,
    SageMakerVLLMEndpointClient,
)
from backend.inference.multimodal import normalize_generate_request
from backend.inference.schemas import GenerateRequest


def test_model_generation_request_round_trip():
    request = ModelGenerationRequest.from_dict(
        {
            "request_id": "req-1",
            "session_id": "sess-1",
            "branch_id": "branch-a",
            "revision_index": 2,
            "parent_request_id": "req-0",
            "prompt": "draw a diagram",
            "feedback": "tighten routing",
            "feedback_rating": 4,
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "width": 640,
            "height": 360,
            "max_tokens": 1024,
        }
    )

    payload = request.to_dict()

    assert payload["session_id"] == "sess-1"
    assert payload["branch_id"] == "branch-a"
    assert payload["revision_index"] == 2
    assert payload["feedback"] == "tighten routing"


def test_sagemaker_vllm_endpoint_client_parses_json_response():
    class _Body:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return self.payload

    class _FakeRuntime:
        def __init__(self):
            self.calls = []

        def invoke_endpoint(self, **kwargs):
            self.calls.append(kwargs)
            body = json.dumps({"request_id": "req-1", "svg": "<svg/>", "model": "Qwen/Qwen2.5-7B-Instruct"}).encode()
            return {"Body": _Body(body)}

    runtime = _FakeRuntime()
    client = SageMakerVLLMEndpointClient(endpoint_name="svg-finetuning-inference", runtime=runtime)
    result = client.generate({"request_id": "req-1", "prompt": "draw a diagram"})

    assert runtime.calls[0]["EndpointName"] == "svg-finetuning-inference"
    assert result["svg"] == "<svg/>"


def test_generate_request_allows_media_inputs_and_normalizes_prompt():
    request = GenerateRequest.from_dict(
        {
            "request_id": "req-1",
            "prompt": "",
            "input_mode": "image",
            "reference_images": ["s3://bucket/reference.png"],
            "media_metadata": {
                "s3://bucket/reference.png": {
                    "filename": "reference.png",
                    "mime_type": "image/png",
                    "width": 1200,
                    "height": 800,
                    "caption": "network architecture sketch",
                }
            },
        }
    )

    result = normalize_generate_request(request)

    assert request.prompt.startswith("Diagram description:")
    assert "Reference media summary:" in request.prompt
    assert "reference.png" in request.prompt
    assert request.diagram_description.startswith("Reference media summary:")
    assert result.original_prompt == ""


def test_generate_request_polishes_text_only_prompts():
    request = GenerateRequest.from_dict(
        {
            "request_id": "req-text-1",
            "prompt": "  make a sequence diagram of auth flow  ",
        }
    )

    result = normalize_generate_request(request)

    assert "User request:" in request.prompt
    assert "Task:" in request.prompt
    assert result.original_prompt == "make a sequence diagram of auth flow"
    assert result.diagram_description == ""


def test_sagemaker_media_description_endpoint_client_parses_json_response():
    class _Body:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return self.payload

    class _FakeRuntime:
        def __init__(self):
            self.calls = []

        def invoke_endpoint(self, **kwargs):
            self.calls.append(kwargs)
            body = json.dumps(
                {
                    "diagram_description": "A three-node architecture diagram with left-to-right flow.",
                    "media_summary": ["reference.png; image/png; 1200x800"],
                }
            ).encode()
            return {"Body": _Body(body)}

    runtime = _FakeRuntime()
    client = SageMakerMediaDescriptionEndpointClient(endpoint_name="svg-finetuning-vision", runtime=runtime)
    result = client.describe({"request_id": "req-1", "reference_images": ["s3://bucket/reference.png"]})

    assert runtime.calls[0]["EndpointName"] == "svg-finetuning-vision"
    assert result["diagram_description"].startswith("A three-node architecture diagram")
