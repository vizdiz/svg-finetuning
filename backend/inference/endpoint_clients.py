"""
backend.inference.endpoint_clients

SageMaker runtime adapters for the internal vLLM contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os
import re
import json
import uuid
import xml.etree.ElementTree as ET

import boto3

from backend.inference.contracts import ModelGenerationRequest
from backend.training.diagram_compiler import compile_diagram_ir
from backend.training.diagram_ir import CanvasSpec, DiagramIRDocument, EdgeSpec, LayoutSpec, NodeSpec


DEFAULT_REPAIR_MODEL_ID = "us.amazon.nova-lite-v1:0"


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    stripped = stripped.replace("<|im_end|>", "").strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _extract_svg_text(text: str) -> str:
    stripped = _strip_code_fences(text)
    match = re.search(r"<svg\b[\s\S]*?</svg>", stripped, flags=re.IGNORECASE)
    return match.group(0).strip() if match else stripped


def _build_lmi_prompt(request: ModelGenerationRequest) -> str:
    normalized_prompt = ""
    original_prompt = ""
    if isinstance(request.metadata, dict):
        original_prompt = str(request.metadata.get("original_prompt") or "").strip()
        normalized_prompt = str(request.metadata.get("normalized_prompt") or "").strip()
    prompt_text = original_prompt or request.prompt.strip() or request.diagram_description.strip() or normalized_prompt
    if not prompt_text:
        prompt_text = "Generate a clean SVG diagram."

    user_parts = [prompt_text]
    if request.diagram_description.strip():
        user_parts.append(f"Diagram description:\n{request.diagram_description.strip()}")
    if request.feedback.strip():
        user_parts.append(f"Revision feedback:\n{request.feedback.strip()}")
    user_prompt = "\n\n".join(part for part in user_parts if part)

    # The LoRA adapter was trained with Qwen's chat template. LMI/vLLM text
    # generation accepts a raw string, so we render the same minimal template
    # here instead of sending instruction prose that the model never saw.
    return (
        "<|im_start|>system\n"
        "You are a diagram compiler. Given a description, produce valid diagram IR JSON only."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _parse_lmi_generation(parsed: Any) -> str:
    if isinstance(parsed, str):
        return _extract_svg_text(parsed)
    if isinstance(parsed, list) and parsed:
        return _parse_lmi_generation(parsed[0])
    if isinstance(parsed, dict):
        for key in ("svg", "generated_text", "generation", "text", "output_text"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return _extract_svg_text(value)
        choices = parsed.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return _extract_svg_text(message["content"])
                if isinstance(first.get("text"), str):
                    return _extract_svg_text(first["text"])
    raise ValueError("endpoint response did not include generated SVG text")


def _message_text(parsed: Any) -> str:
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        if isinstance(parsed.get("content"), list):
            parts: list[str] = []
            for block in parsed["content"]:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "".join(parts).strip()
        for key in ("text", "output_text", "completion"):
            if isinstance(parsed.get(key), str):
                return parsed[key].strip()
    return ""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = _strip_code_fences(text)
    candidates = [stripped]
    repaired = re.sub(r',\s*"\{', ",{", stripped)
    if repaired != stripped:
        candidates.append(repaired)

    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    for candidate in candidates:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def _slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or fallback


def _compile_text_ir(text: str) -> str | None:
    stripped = _strip_code_fences(text)
    entities: list[str] = []
    relationships: list[tuple[str, str]] = []
    layout = "horizontal"
    current_section = ""

    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key, _, value = line.partition(":")
        lower_key = key.strip().lower()
        if lower_key in {"entities", "nodes"}:
            current_section = "entities"
            if value.strip():
                entities.extend(part.strip(" -") for part in value.split(",") if part.strip(" -"))
            continue
        if lower_key in {"relationships", "edges"}:
            current_section = "relationships"
            if value.strip():
                for part in value.split(","):
                    if "->" in part:
                        source, target = part.split("->", 1)
                        relationships.append((source.strip(), target.strip()))
            continue
        if lower_key == "layout":
            if "vertical" in value.lower():
                layout = "vertical"
            elif "horizontal" in value.lower() or "left-to-right" in value.lower():
                layout = "horizontal"
            continue
        if current_section == "entities":
            entities.append(line.strip(" -"))
        elif current_section == "relationships" and "->" in line:
            source, target = line.strip(" -").split("->", 1)
            relationships.append((source.strip(), target.strip()))

    cleaned_entities: list[str] = []
    for entity in entities:
        entity = re.sub(r"\s*\([^)]*\)\s*", "", entity).strip()
        if entity and entity not in cleaned_entities:
            cleaned_entities.append(entity)
    if len(cleaned_entities) < 2:
        return None

    id_by_label = {
        label: _slug(label, f"node-{index + 1}")
        for index, label in enumerate(cleaned_entities[:8])
    }
    nodes = [
        NodeSpec(id=node_id, kind="box", label=label)
        for label, node_id in id_by_label.items()
    ]

    edges: list[EdgeSpec] = []
    for source_label, target_label in relationships:
        source_id = id_by_label.get(source_label)
        target_id = id_by_label.get(target_label)
        if source_id and target_id and source_id != target_id:
            edges.append(
                EdgeSpec(
                    id=f"edge-{len(edges) + 1}",
                    source=source_id,
                    target=target_id,
                    directed=True,
                )
            )
    if not edges:
        node_ids = [node.id for node in nodes]
        for index, (source_id, target_id) in enumerate(zip(node_ids, node_ids[1:]), start=1):
            edges.append(EdgeSpec(id=f"edge-{index}", source=source_id, target=target_id, directed=True))

    return compile_diagram_ir(
        DiagramIRDocument(
            diagram_type="flowchart",
            title="svgen",
            canvas=CanvasSpec(width=640, height=480, padding=64),
            nodes=nodes,
            edges=edges,
            layout=LayoutSpec(direction=layout, spacing=72),
            metadata={"compiled_from": "text_ir"},
        )
    )


def _compile_graph_json(payload: dict[str, Any]) -> str | None:
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list) or len(raw_nodes) < 2:
        return None

    nodes: list[NodeSpec] = []
    for index, raw_node in enumerate(raw_nodes[:8], start=1):
        if not isinstance(raw_node, dict):
            continue
        data = raw_node.get("data") if isinstance(raw_node.get("data"), dict) else {}
        label = str(raw_node.get("label") or data.get("label") or raw_node.get("id") or f"node {index}")
        node_id = _slug(str(raw_node.get("id") or label), f"node-{index}")
        nodes.append(NodeSpec(id=node_id, kind="box", label=label))

    if len(nodes) < 2:
        return None

    node_ids = {node.id for node in nodes}
    raw_edges = payload.get("edges") if isinstance(payload.get("edges"), list) else []
    edges: list[EdgeSpec] = []
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, dict):
            continue
        data = raw_edge.get("data") if isinstance(raw_edge.get("data"), dict) else {}
        source = _slug(str(raw_edge.get("source") or data.get("source") or ""), "")
        target = _slug(str(raw_edge.get("target") or data.get("target") or ""), "")
        if source in node_ids and target in node_ids and source != target:
            edges.append(EdgeSpec(id=f"edge-{len(edges) + 1}", source=source, target=target, directed=True))

    if not edges:
        ordered_ids = [node.id for node in nodes]
        for source, target in zip(ordered_ids, ordered_ids[1:]):
            edges.append(EdgeSpec(id=f"edge-{len(edges) + 1}", source=source, target=target, directed=True))

    title = str(payload.get("title") or "svgen")
    return compile_diagram_ir(
        DiagramIRDocument(
            diagram_type="flowchart",
            title=title,
            canvas=CanvasSpec(width=640, height=480, padding=64),
            nodes=nodes,
            edges=edges,
            layout=LayoutSpec(direction="horizontal", spacing=72),
            metadata={"compiled_from": "graph_json"},
        )
    )


def _compile_loose_model_ir(text: str) -> str | None:
    stripped = _strip_code_fences(text)
    node_matches = re.findall(
        r'\{[^{}]*"group_id"\s*:\s*[^{}]*?"id"\s*:\s*"([^"]+)"[^{}]*?'
        r'"kind"\s*:\s*"([^"]+)"[^{}]*?"label"\s*:\s*"([^"]*)"',
        stripped,
    )
    nodes: list[NodeSpec] = []
    for index, (node_id, kind, label) in enumerate(node_matches[:8], start=1):
        clean_id = _slug(node_id, f"node-{index}")
        if clean_id and clean_id not in {node.id for node in nodes}:
            nodes.append(NodeSpec(id=clean_id, kind=kind or "box", label=label or node_id))

    if len(nodes) < 2:
        simple_labels = [
            value
            for value in re.findall(r'"label"\s*:\s*"([^"]+)"', stripped)
            if not value.startswith("edge-")
        ]
        for index, label in enumerate(simple_labels[:8], start=1):
            clean_id = _slug(label, f"node-{index}")
            if clean_id and clean_id not in {node.id for node in nodes}:
                nodes.append(NodeSpec(id=clean_id, kind="box", label=label))

    if len(nodes) < 2:
        return None

    node_ids = {node.id for node in nodes}
    edge_matches = re.findall(
        r'"label"\s*:\s*"([^"]*)"[\s\S]{0,400}?"source"\s*:\s*"([^"]+)"'
        r'[\s\S]{0,400}?"target"\s*:\s*"([^"]+)"',
        stripped,
    )
    edges: list[EdgeSpec] = []
    for label, source, target in edge_matches:
        source_id = _slug(source, "")
        target_id = _slug(target, "")
        if source_id in node_ids and target_id in node_ids and source_id != target_id:
            edges.append(
                EdgeSpec(
                    id=f"edge-{len(edges) + 1}",
                    source=source_id,
                    target=target_id,
                    directed=True,
                    label=label,
                    routing="straight",
                )
            )

    if not edges:
        ordered_ids = [node.id for node in nodes]
        for source, target in zip(ordered_ids, ordered_ids[1:]):
            edges.append(EdgeSpec(id=f"edge-{len(edges) + 1}", source=source, target=target, directed=True))

    title_match = re.search(r'"title"\s*:\s*"([^"]+)"', stripped)
    title = title_match.group(1) if title_match else "svgen"
    return compile_diagram_ir(
        DiagramIRDocument(
            diagram_type="sequence" if "sequence" in stripped.lower() else "flowchart",
            title=title,
            canvas=CanvasSpec(width=640, height=480, padding=64),
            nodes=nodes,
            edges=edges,
            layout=LayoutSpec(direction="horizontal", spacing=72, edge_routing="straight"),
            metadata={"compiled_from": "loose_model_ir"},
        )
    )


def _compile_ir_generation(text: str, *, allow_loose: bool = True) -> str | None:
    payload = _extract_json_object(text)
    if payload is not None:
        try:
            return compile_diagram_ir(DiagramIRDocument.from_dict(payload))
        except Exception:
            compiled = _compile_graph_json(payload)
            if compiled:
                return compiled
            raise
    if allow_loose:
        compiled = _compile_loose_model_ir(text)
        if compiled:
            return compiled
    return _compile_text_ir(text)


def _ensure_svg(text: str, *, allow_loose: bool = True) -> tuple[str, str]:
    if re.search(r"<svg\b", text, flags=re.IGNORECASE):
        try:
            ET.fromstring(text)
            return text, "svg"
        except ET.ParseError as exc:
            raise ValueError("model returned malformed svg") from exc
    compiled = _compile_ir_generation(text, allow_loose=allow_loose)
    if compiled:
        return compiled, "diagram_ir"
    raise ValueError("model returned neither svg nor compilable diagram ir")


def _bedrock_repair_prompt(request: ModelGenerationRequest, raw_generation: str) -> str:
    prompt_text = request.prompt.strip()
    if isinstance(request.metadata, dict):
        prompt_text = str(request.metadata.get("original_prompt") or prompt_text).strip()
    if request.diagram_description.strip():
        prompt_text = f"{prompt_text}\n\nDiagram description:\n{request.diagram_description.strip()}".strip()
    if request.feedback.strip():
        prompt_text = f"{prompt_text}\n\nRevision feedback:\n{request.feedback.strip()}".strip()
    return (
        "Repair a malformed diagram model output into valid canonical diagram IR JSON only.\n"
        "Do not return Markdown, SVG, prose, or code fences.\n"
        "Use this schema shape and only compatible fields:\n"
        '{"schema_version":"0.1","diagram_type":"flowchart","title":"...",'
        '"canvas":{"width":640,"height":480,"grid":8,"origin_x":0,"origin_y":0,"padding":64,"background":"#F0EFEB"},'
        '"layout":{"direction":"horizontal","alignment":"center","spacing":72,"snap_grid":8,"text_wrap":"balanced","edge_routing":"straight"},'
        '"nodes":[{"id":"node-id","kind":"box","label":"Label"}],'
        '"edges":[{"id":"edge-1","source":"source-id","target":"target-id","directed":true,"label":"","routing":"straight"}],'
        '"groups":[]}\n\n'
        "Requirements:\n"
        "- Include all important entities from the user prompt.\n"
        "- Edges must reference existing node ids.\n"
        "- Use lowercase kebab-case node ids.\n"
        "- Omit ports unless they are explicitly defined on nodes.\n"
        "- Keep the diagram concise enough for a 640x480 canvas.\n\n"
        f"User prompt:\n{prompt_text}\n\n"
        f"Malformed model output:\n{raw_generation}"
    )


def _repair_generation_with_bedrock(
    request: ModelGenerationRequest,
    raw_generation: str,
    *,
    runtime: Any | None = None,
    region: str = "us-east-1",
    model_id: str = DEFAULT_REPAIR_MODEL_ID,
) -> str:
    runtime = runtime or boto3.client("bedrock-runtime", region_name=region)
    prompt = _bedrock_repair_prompt(request, raw_generation)
    if model_id.startswith(("us.amazon.", "global.amazon.", "amazon.")):
        response = runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 1200, "temperature": 0.2},
        )
        repaired = "".join(
            block.get("text", "")
            for block in response.get("output", {}).get("message", {}).get("content", [])
            if isinstance(block, dict)
        ).strip()
        if not repaired:
            raise ValueError("repair model returned empty response")
        return _strip_code_fences(repaired)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1200,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    response = runtime.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body).encode(),
    )
    parsed = json.loads(response["body"].read().decode())
    repaired = _message_text(parsed)
    if not repaired:
        raise ValueError("repair model returned empty response")
    return _strip_code_fences(repaired)


def _bedrock_baseline_prompt(request: ModelGenerationRequest) -> str:
    prompt_text = request.prompt.strip()
    if isinstance(request.metadata, dict):
        prompt_text = str(request.metadata.get("original_prompt") or prompt_text).strip()
    if request.diagram_description.strip():
        prompt_text = f"{prompt_text}\n\nDiagram description:\n{request.diagram_description.strip()}".strip()
    if request.feedback.strip():
        prompt_text = f"{prompt_text}\n\nRevision feedback:\n{request.feedback.strip()}".strip()
    return (
        "Generate a complete, well-formed SVG technical diagram only. "
        "Do not return Markdown, prose, code fences, JSON, or explanations.\n\n"
        "Rules:\n"
        "- Include all important entities and named services from the prompt.\n"
        "- Use 4 to 8 nodes for non-trivial workflows.\n"
        "- Use a 640 by 480 viewBox.\n"
        "- Use simple labeled boxes, readable text, and connector arrows.\n"
        "- Keep labels inside boxes or centered above connector lines; do not overlap text.\n"
        "- Use these colors only: #F0EFEB, #E8E7E2, #1A1A18, #D0CFC8, #F59E0B.\n"
        "- Use IBM Plex Mono as the font-family.\n\n"
        f"User prompt:\n{prompt_text}"
    )


def _invoke_bedrock_text(
    prompt: str,
    *,
    runtime: Any,
    model_id: str,
    max_tokens: int,
) -> str:
    if model_id.startswith(("us.amazon.", "global.amazon.", "amazon.")):
        response = runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        return "".join(
            block.get("text", "")
            for block in response.get("output", {}).get("message", {}).get("content", [])
            if isinstance(block, dict)
        ).strip()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = runtime.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body).encode(),
    )
    parsed = json.loads(response["body"].read().decode())
    return _message_text(parsed)


@dataclass
class SageMakerVLLMEndpointClient:
    endpoint_name: str
    region: str = "us-east-1"
    timeout_s: float = 60.0
    runtime: Any | None = None
    repair_runtime: Any | None = None

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        if not str(payload.get("request_id") or "").strip():
            payload["request_id"] = uuid.uuid4().hex
        request = ModelGenerationRequest.from_dict(payload)
        runtime = self.runtime or boto3.client("sagemaker-runtime", region_name=self.region)
        body = {
            "inputs": _build_lmi_prompt(request),
            "parameters": {
                "temperature": float(request.temperature or 0.0),
                "max_tokens": int(request.max_tokens or 512),
                "max_new_tokens": int(request.max_tokens or 512),
                "stop": ["<|im_end|>"],
            },
        }
        response = runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(body).encode(),
        )
        raw_body = response["Body"].read()
        try:
            parsed = json.loads(raw_body.decode())
        except Exception:
            parsed = raw_body.decode()

        raw_generation = _parse_lmi_generation(parsed)
        repair_metadata: dict[str, Any] = {}
        try:
            svg, output_format = _ensure_svg(raw_generation, allow_loose=False)
        except ValueError as original_exc:
            if os.environ.get("BEDROCK_REPAIR_ENABLED", "false").lower() != "true":
                svg, output_format = _ensure_svg(raw_generation, allow_loose=True)
            else:
                repair_model_id = os.environ.get("BEDROCK_REPAIR_MODEL_ID", DEFAULT_REPAIR_MODEL_ID)
                try:
                    repaired_generation = _repair_generation_with_bedrock(
                        request,
                        raw_generation,
                        runtime=self.repair_runtime,
                        region=self.region,
                        model_id=repair_model_id,
                    )
                    svg, output_format = _ensure_svg(repaired_generation, allow_loose=False)
                    output_format = f"bedrock_repaired_{output_format}"
                    repair_metadata = {
                        "repair_model": repair_model_id,
                        "repaired_generation": repaired_generation,
                    }
                except Exception as repair_exc:
                    if os.environ.get("BEDROCK_REPAIR_ALLOW_LOOSE_FALLBACK", "false").lower() == "true":
                        try:
                            svg, output_format = _ensure_svg(raw_generation, allow_loose=True)
                            output_format = f"loose_repaired_{output_format}"
                            repair_metadata = {
                                "repair_error": str(repair_exc),
                                "strict_error": str(original_exc),
                            }
                        except Exception:
                            raise original_exc
                    else:
                        raise ValueError(f"repair model failed: {repair_exc}") from repair_exc

        return {
            "svg": svg,
            "request_id": request.request_id,
            "model": request.model,
            "metadata": {
                "endpoint": self.endpoint_name,
                "payload_format": "lmi-text-generation",
                "output_format": output_format,
                "raw_generation": raw_generation,
                **repair_metadata,
            },
        }


@dataclass
class BedrockBaselineEndpointClient:
    model_id: str = DEFAULT_REPAIR_MODEL_ID
    region: str = "us-east-1"
    runtime: Any | None = None

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        if not str(payload.get("request_id") or "").strip():
            payload["request_id"] = uuid.uuid4().hex
        request = ModelGenerationRequest.from_dict(payload)
        runtime = self.runtime or boto3.client("bedrock-runtime", region_name=self.region)
        raw_generation = _invoke_bedrock_text(
            _bedrock_baseline_prompt(request),
            runtime=runtime,
            model_id=self.model_id,
            max_tokens=max(800, min(int(request.max_tokens or 1200), 1800)),
        )
        raw_generation = _strip_code_fences(raw_generation)
        svg, output_format = _ensure_svg(raw_generation, allow_loose=False)
        return {
            "svg": svg,
            "request_id": request.request_id,
            "model": self.model_id,
            "metadata": {
                "endpoint": "bedrock",
                "payload_format": "bedrock-baseline-ir",
                "output_format": output_format,
                "raw_generation": raw_generation,
            },
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
