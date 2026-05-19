"""
dataset_pipeline.config

Central configuration for the dataset pipeline.

Loads all settings from environment variables (via python-dotenv) with
sensible defaults. No hardcoded account IDs, bucket names, or API keys
— everything is driven by the .env file or the shell environment.

Settings include:
  - AWS: S3 bucket names, region, profile
  - Anthropic: API key, model to use for captioning, max tokens
  - Scraper limits: max records per source, request timeouts, retry counts
  - Validation thresholds: min/max SVG byte size, max raster image ratio
  - Normalizer: decimal precision for coordinate rounding
  - Captioner: prompt template, min/max caption length, cache behaviour
  - Pipeline: batch ID prefix, parallelism (thread pool size)
"""
