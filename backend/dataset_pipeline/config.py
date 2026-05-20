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

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class PipelineConfig:
    anthropic_api_key: str = field(default_factory=lambda: os.environ["ANTHROPIC_API_KEY"])
    aws_profile: str = field(default_factory=lambda: os.environ["AWS_PROFILE"])
    s3_data_bucket: str = field(default_factory=lambda: os.environ["S3_DATA_BUCKET"])
    s3_scripts_bucket: str = field(default_factory=lambda: os.environ["S3_SCRIPTS_BUCKET"])
    region: str = field(default_factory=lambda: os.environ["REGION"])
    caption_backend: str = "anthropic"
    caption_local_base_url: str = "http://localhost:11434/v1"
    caption_local_model: str = "llava"
    caption_local_timeout_s: float = 60.0
    raw_prefix: str = "raw/"
    processed_prefix: str = "processed/"
    train_prefix: str = "train/"
    val_prefix: str = "val/"


@dataclass
class ScrapeConfig:
    max_svgs_per_source: int = 1000
    min_svg_bytes: int = 500
    max_svg_bytes: int = 5_000_000
    val_split: float = 0.1
    min_svg_elements: int = 5
    max_svg_nodes: int = 50_000
    max_total_path_chars: int = 2_000_000
    max_single_attr_chars: int = 500_000
    hard_min_aspect_ratio: float = 0.02
    hard_max_aspect_ratio: float = 50.0
    warning_min_aspect_ratio: float = 0.1
    warning_max_aspect_ratio: float = 10.0
    quality_report_every: int = 1000
    wikimedia_cache_dir: str = ".cache/wikimedia_svgs"
    wikimedia_download_delay_s: float = 5.0
    wikimedia_max_consecutive_429: int = 3
    wikimedia_429_cooldown_s: float = 60.0
    wikimedia_dump_path: str = ""
    wikimedia_asset_base_url: str = "https://upload.wikimedia.org/wikipedia/commons"
    wikimedia_dump_download_parallelism: int = 8
    wikimedia_dump_max_inflight: int = 64
    wikimedia_dump_rank_multiplier: int = 6
    caption_parallelism: int = 4
    retrieval_parallelism: int = 2
    caption_backend: str = "anthropic"
    caption_local_base_url: str = "http://localhost:11434/v1"
    caption_local_model: str = "llava"
    caption_local_timeout_s: float = 60.0
    dump_ir: bool = False
