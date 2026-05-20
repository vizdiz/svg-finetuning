FROM vllm/vllm-openai:latest

ENV MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
ENV HOST=0.0.0.0
ENV PORT=8000
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.90

EXPOSE 8000

CMD ["/bin/sh", "-lc", "python -m uvicorn backend.inference.vllm.server:app --host \"$HOST\" --port \"$PORT\""]
