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
    return (
        f"{request.system_prompt}\n\n"
        f"Prompt:\n{request.prompt}\n\n"
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
        },
    )
    return asdict(response)
