from __future__ import annotations

import json

from backend.inference.contracts import ModelGenerationRequest
from backend.inference.endpoint_clients import SageMakerVLLMEndpointClient


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
