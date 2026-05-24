from __future__ import annotations

import json
import pytest

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
            body = json.dumps([{"generated_text": "```svg\n<svg/>\n```"}]).encode()
            return {"Body": _Body(body)}

    runtime = _FakeRuntime()
    client = SageMakerVLLMEndpointClient(endpoint_name="svg-finetuning-inference", runtime=runtime)
    result = client.generate({"request_id": "req-1", "prompt": "draw a diagram"})

    assert runtime.calls[0]["EndpointName"] == "svg-finetuning-inference"
    sent = json.loads(runtime.calls[0]["Body"].decode())
    assert "inputs" in sent
    assert sent["parameters"]["max_tokens"] == 512
    assert sent["parameters"]["max_new_tokens"] == 512
    assert sent["inputs"].startswith("<|im_start|>system")
    assert result["svg"] == "<svg/>"
    assert result["request_id"]


def test_sagemaker_vllm_endpoint_client_compiles_text_ir_response():
    class _Body:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return self.payload

    class _FakeRuntime:
        def invoke_endpoint(self, **kwargs):
            body = json.dumps([{"generated_text": "Entities: input, model, svg"}]).encode()
            return {"Body": _Body(body)}

    client = SageMakerVLLMEndpointClient(endpoint_name="svg-finetuning-inference", runtime=_FakeRuntime())

    result = client.generate({"request_id": "req-1", "prompt": "draw a diagram"})

    assert result["svg"].startswith("<svg")
    assert 'id="node-input"' in result["svg"]
    assert 'id="node-model"' in result["svg"]
    assert result["metadata"]["output_format"] == "diagram_ir"


def test_sagemaker_vllm_endpoint_client_compiles_graph_json_response():
    class _Body:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return self.payload

    class _FakeRuntime:
        def invoke_endpoint(self, **kwargs):
            generated = (
                '{"nodes":[{"id":"n0","data":{"label":"input","shape":"rectangle"}},'
                '"{"id":"n1","data":{"label":"model","shape":"rectangle"}},'
                '"{"id":"n2","data":{"label":"svg","shape":"rectangle"}}],'
                '"edges":[{"data":{"source":"n0","target":"n1"}},'
                '"{"source":"n1","target":"n2"}]}<|im_end|>'
            )
            body = json.dumps([{"generated_text": generated}]).encode()
            return {"Body": _Body(body)}

    client = SageMakerVLLMEndpointClient(endpoint_name="svg-finetuning-inference", runtime=_FakeRuntime())

    result = client.generate({"request_id": "req-1", "prompt": "draw a diagram"})

    assert result["svg"].startswith("<svg")
    assert ">input<" in result["svg"]
    assert ">model<" in result["svg"]
    assert ">svg<" in result["svg"]
    assert result["metadata"]["output_format"] == "diagram_ir"


def test_sagemaker_vllm_endpoint_client_repairs_loose_strict_ir_response():
    class _Body:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return self.payload

    class _FakeRuntime:
        def invoke_endpoint(self, **kwargs):
            generated = (
                '{"canvas":{"width":640},"diagram_type":"sequence","edges":[{'
                '"directed":true,"id":"edge-1","label":"","source":"user","target":"spotify"},'
                '"edges":[{"directed":true,"id":"edge-2","label":"access_token",'
                '"source":"spotify","target":"user"}],"nodes":[{"group_id":null,'
                '"id":"user","kind":"actor","label":"user","style":{"text_weight":null}},'
                '{"group_id":null,"id":"spotify","kind":"service","label":"Spotify",'
                '"type":null}],"title":"User auth flow for Spotify"}<|im_end|>'
            )
            body = json.dumps([{"generated_text": generated}]).encode()
            return {"Body": _Body(body)}

    client = SageMakerVLLMEndpointClient(endpoint_name="svg-finetuning-inference", runtime=_FakeRuntime())

    result = client.generate({"request_id": "req-1", "prompt": "auth flow"})

    assert result["svg"].startswith("<svg")
    assert ">user<" in result["svg"]
    assert ">Spotify<" in result["svg"]
    assert "access_token" in result["svg"]
    assert result["metadata"]["output_format"] == "diagram_ir"


def test_sagemaker_vllm_endpoint_client_rejects_malformed_svg_response():
    class _Body:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return self.payload

    class _FakeRuntime:
        def invoke_endpoint(self, **kwargs):
            body = json.dumps([{"generated_text": '<svg fill width="10"></svg>'}]).encode()
            return {"Body": _Body(body)}

    client = SageMakerVLLMEndpointClient(endpoint_name="svg-finetuning-inference", runtime=_FakeRuntime())

    with pytest.raises(ValueError, match="model returned malformed svg"):
        client.generate({"request_id": "req-1", "prompt": "draw a diagram"})


def test_sagemaker_vllm_endpoint_client_rejects_uncompilable_response():
    class _Body:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return self.payload

    class _FakeRuntime:
        def invoke_endpoint(self, **kwargs):
            body = json.dumps([{"generated_text": "not a diagram"}]).encode()
            return {"Body": _Body(body)}

    client = SageMakerVLLMEndpointClient(endpoint_name="svg-finetuning-inference", runtime=_FakeRuntime())

    with pytest.raises(ValueError, match="neither svg nor compilable diagram ir"):
        client.generate({"request_id": "req-1", "prompt": "draw a diagram"})


def test_sagemaker_vllm_endpoint_client_can_use_bedrock_repair(monkeypatch):
    class _Body:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return self.payload

    class _FakeRuntime:
        def invoke_endpoint(self, **kwargs):
            body = json.dumps([{"generated_text": '{"nodes":[malformed'}]).encode()
            return {"Body": _Body(body)}

    class _FakeRepairRuntime:
        def __init__(self):
            self.calls = []

        def converse(self, **kwargs):
            self.calls.append(kwargs)
            repaired = {
                "schema_version": "0.1",
                "diagram_type": "flowchart",
                "title": "repaired",
                "canvas": {"width": 640, "height": 480, "padding": 64},
                "layout": {"direction": "horizontal", "spacing": 72, "edge_routing": "straight"},
                "nodes": [
                    {"id": "input", "kind": "box", "label": "input"},
                    {"id": "model", "kind": "box", "label": "model"},
                ],
                "edges": [
                    {"id": "edge-1", "source": "input", "target": "model", "directed": True},
                ],
                "groups": [],
            }
            return {
                "output": {
                    "message": {
                        "content": [{"text": json.dumps(repaired)}],
                    }
                }
            }

        def invoke_model(self, **kwargs):
            self.calls.append(kwargs)
            repaired = {
                "schema_version": "0.1",
                "diagram_type": "flowchart",
                "title": "repaired",
                "canvas": {"width": 640, "height": 480, "padding": 64},
                "layout": {"direction": "horizontal", "spacing": 72, "edge_routing": "straight"},
                "nodes": [
                    {"id": "input", "kind": "box", "label": "input"},
                    {"id": "model", "kind": "box", "label": "model"},
                ],
                "edges": [
                    {"id": "edge-1", "source": "input", "target": "model", "directed": True},
                ],
                "groups": [],
            }
            body = json.dumps(
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(repaired),
                        }
                    ]
                }
            ).encode()
            return {"body": _Body(body)}

    monkeypatch.setenv("BEDROCK_REPAIR_ENABLED", "true")
    repair_runtime = _FakeRepairRuntime()
    client = SageMakerVLLMEndpointClient(
        endpoint_name="svg-finetuning-inference",
        runtime=_FakeRuntime(),
        repair_runtime=repair_runtime,
    )

    result = client.generate({"request_id": "req-1", "prompt": "draw input to model"})

    assert result["svg"].startswith("<svg")
    assert result["metadata"]["output_format"] == "bedrock_repaired_diagram_ir"
    assert repair_runtime.calls


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
