"""
dataset_pipeline.pipeline.runner

Top-level orchestrator for a dataset pipeline run.

Instantiates the configured scrapers, feeds each RawSVGRecord through the
processing stages (validate → normalise → caption), and uploads completed
TrainingRecords as JSONL to s3://<data-bucket>/train/<batch-id>/.

Progress is tracked via tqdm with per-scraper counters for total seen,
passed validation, and dropped (with breakdown by rejection reason).

On completion, calls manifest_writer to finalise the batch. If any stage
fails fatally, the partial batch is abandoned and no manifest is written
(preventing a malformed dataset from triggering training).

Configurable via config.py and environment variables; intended to be run
as a one-off script locally or as a scheduled ECS/Batch job on AWS.
"""
