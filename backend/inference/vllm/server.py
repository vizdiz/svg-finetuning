"""
Custom SageMaker vLLM server.

This endpoint accepts the repo's internal JSON contract and returns SVG
only. It is deliberately smaller than an OpenAI-compatible surface:
the public API already owns request routing, cache keys, sessions, and
revision lineage.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import os

from fastapi import FastAPI, Request

from backend.inference.contracts import ModelGenerationRequest, ModelGenerationResponse

app = FastAPI(title="svgen-vllm", version="1.0.0")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


@lru_cache(maxsize=1)
def _load_model():
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "4096"))
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # pragma: no cover - only used in the container
        raise RuntimeError(f"vLLM is unavailable: {exc}") from exc
    return LLM(model=model_name, max_model_len=max_model_len), SamplingParams(temperature=0.0, max_tokens=2048)


def _build_prompt(request: ModelGenerationRequest) -> str:
    normalized_prompt = ""
    if isinstance(request.metadata, dict):
        normalized_prompt = str(request.metadata.get("normalized_prompt") or "").strip()
    prompt_text = normalized_prompt or request.prompt.strip() or request.diagram_description.strip() or "Describe the uploaded reference media."
    diagram_description = request.diagram_description.strip()
    media_summary = request.metadata.get("media_summary") if isinstance(request.metadata, dict) else None
    media_block = ""
    if diagram_description:
        media_block = f"\nDiagram description:\n{diagram_description}\n"
    elif media_summary:
        if isinstance(media_summary, list):
            media_text = "\n".join(f"- {item}" for item in media_summary if str(item).strip())
        else:
            media_text = str(media_summary)
        media_block = f"\nDiagram description:\n{media_text}\n"
    return (
        f"{request.system_prompt}\n\n"
        f"Prompt:\n{prompt_text}\n\n"
        f"{media_block}"
        f"Revision index: {request.revision_index}\n"
        f"Branch id: {request.branch_id}\n"
        f"Feedback: {request.feedback}\n"
        f"Return SVG only.\n"
    )


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/invocations")
async def invocations(request: Request) -> dict[str, object]:
    payload = await request.json()
    model_request = ModelGenerationRequest.from_dict(payload)

    try:
        llm, sampling_params = _load_model()
        prompt = _build_prompt(model_request)
        outputs = llm.generate([prompt], sampling_params=sampling_params)
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        svg = _strip_code_fences(text)
    except Exception:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{model_request.width}" height="{model_request.height}">'
            f'<rect width="100%" height="100%" fill="#E8E7E2"/>'
            f'<text x="20" y="40" font-size="16" font-family="monospace">model unavailable</text>'
            f'<text x="20" y="70" font-size="12" font-family="monospace">request {model_request.request_id}</text>'
            f"</svg>"
        )

    response = ModelGenerationResponse(
        request_id=model_request.request_id,
        svg=svg,
        model=model_request.model or os.environ.get("MODEL_NAME", ""),
        metadata={
            "session_id": model_request.session_id,
            "branch_id": model_request.branch_id,
            "revision_index": model_request.revision_index,
            "parent_request_id": model_request.parent_request_id,
            "feedback_rating": model_request.feedback_rating,
            "input_mode": model_request.input_mode,
            "diagram_description": model_request.diagram_description,
            "reference_images": model_request.reference_images,
        },
    )
    return asdict(response)
