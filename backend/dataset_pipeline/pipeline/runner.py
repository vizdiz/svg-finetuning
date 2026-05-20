"""
dataset_pipeline.pipeline.runner

Main entrypoint for scraping, validating, captioning, and writing the SVG
training dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import boto3

from backend.dataset_pipeline.config import PipelineConfig, ScrapeConfig
from backend.dataset_pipeline.pipeline.manifest_writer import dry_run_write, write_manifest
from backend.dataset_pipeline.processing.ir_labeler import label_raw_svg_ir
from backend.dataset_pipeline.processing.captioner import batch_caption
from backend.dataset_pipeline.processing.validator import SVGValidationResult, svg_quality_score, validate_svg, validate_svg_detailed
from backend.dataset_pipeline.scrapers.arxiv_scraper import ArxivScraper
from backend.dataset_pipeline.scrapers.base_scraper import BaseScraper, RawSVG
from backend.dataset_pipeline.scrapers.wikimedia_dump_scraper import WikimediaDumpScraper
from backend.dataset_pipeline.scrapers.wikipedia_scraper import WikipediaScraper

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    scraped: int = 0
    raster_skipped: int = 0
    failed_validation: int = 0
    failed_ir_validation: int = 0
    failed_caption: int = 0
    written_to_s3: int = 0
    manifest_uri: str = ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SVG dataset pipeline")
    parser.add_argument(
        "--sources",
        choices=("arxiv", "wikipedia", "both"),
        default="both",
        help="Which source scraper(s) to run",
    )
    parser.add_argument("--max-per-source", type=int, default=1000, help="Maximum valid SVGs per source")
    parser.add_argument(
        "--retrieval-parallelism",
        type=int,
        default=2,
        help="Number of source scrapers to run concurrently; Wikimedia remains internally rate-limited",
    )
    parser.add_argument(
        "--caption-parallelism",
        type=int,
        default=4,
        help="Number of concurrent Anthropic caption requests",
    )
    parser.add_argument(
        "--caption-backend",
        choices=("anthropic", "local"),
        default="anthropic",
        help="Caption with Anthropic or a local OpenAI-compatible multimodal endpoint",
    )
    parser.add_argument(
        "--caption-local-base-url",
        default="http://localhost:11434/v1",
        help="Base URL for the local caption endpoint when --caption-backend local",
    )
    parser.add_argument(
        "--caption-local-model",
        default="llava",
        help="Local caption model name when --caption-backend local",
    )
    parser.add_argument(
        "--wikipedia-source",
        choices=("api", "dump"),
        default="api",
        help="Use live Commons API discovery or a local Commons image SQL dump for Wikimedia",
    )
    parser.add_argument(
        "--wikimedia-dump-path",
        default="",
        help="Path to commonswiki-latest-image.sql.gz or another Commons image.sql(.gz) snapshot",
    )
    parser.add_argument(
        "--wikimedia-asset-base-url",
        default="https://upload.wikimedia.org/wikipedia/commons",
        help="Base URL for raw Commons assets; mirrors must preserve the hashed upload path layout",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write local outputs and skip S3/Lambda trigger")
    parser.add_argument(
        "--output-dir",
        default="./pipeline_output",
        help="Local output directory used in dry-run mode",
    )
    parser.add_argument(
        "--no-caption",
        action="store_true",
        help="Skip captioning and output validated raw SVGs only for debugging",
    )
    parser.add_argument(
        "--require-diagram-ir",
        action="store_true",
        help="Require each SVG to round-trip through the diagram IR before it is emitted",
    )
    parser.add_argument(
        "--dump-ir",
        action="store_true",
        help="Write accepted diagram IR JSON next to dry-run outputs for inspection",
    )
    return parser


def _selected_scrapers(source: str, wikipedia_source: str = "api") -> list[BaseScraper]:
    wikipedia_scraper: BaseScraper = WikimediaDumpScraper() if wikipedia_source == "dump" else WikipediaScraper()
    if source == "arxiv":
        return [ArxivScraper()]
    if source == "wikipedia":
        return [wikipedia_scraper]
    return [ArxivScraper(), wikipedia_scraper]


def _load_pipeline_config(args: argparse.Namespace):
    if not args.dry_run:
        config = PipelineConfig()
        config.caption_backend = getattr(args, "caption_backend", getattr(config, "caption_backend", "anthropic"))
        config.caption_local_base_url = getattr(
            args, "caption_local_base_url", getattr(config, "caption_local_base_url", "http://localhost:11434/v1")
        )
        config.caption_local_model = getattr(args, "caption_local_model", getattr(config, "caption_local_model", "llava"))
        config.caption_local_timeout_s = getattr(
            args, "caption_local_timeout_s", getattr(config, "caption_local_timeout_s", 60.0)
        )
        return config

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    caption_backend = getattr(args, "caption_backend", "anthropic")
    if not args.no_caption and caption_backend == "anthropic" and not anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for captioned dry runs")

    return SimpleNamespace(
        anthropic_api_key=anthropic_api_key,
        aws_profile=os.environ.get("AWS_PROFILE", ""),
        region=os.environ.get("REGION", os.environ.get("AWS_REGION", "us-east-1")),
        s3_data_bucket=os.environ.get("S3_DATA_BUCKET", "dry-run"),
        raw_prefix="raw/",
        caption_parallelism=getattr(args, "caption_parallelism", 4),
        caption_backend=caption_backend,
        caption_local_base_url=getattr(args, "caption_local_base_url", "http://localhost:11434/v1"),
        caption_local_model=getattr(args, "caption_local_model", "llava"),
        caption_local_timeout_s=60.0,
    )


def _raw_s3_key(raw_svg: RawSVG, config: PipelineConfig) -> str:
    return f"{config.raw_prefix}{raw_svg.domain}/{raw_svg.source_id}.svg"


def _upload_raw_svg(raw_svg: RawSVG, config: PipelineConfig, s3_client) -> bool:
    key = _raw_s3_key(raw_svg, config)
    for attempt in range(2):
        try:
            s3_client.put_object(
                Bucket=config.s3_data_bucket,
                Key=key,
                Body=raw_svg.svg_string.encode(),
                ContentType="image/svg+xml",
            )
            logger.info("Uploaded s3://%s/%s", config.s3_data_bucket, key)
            return True
        except Exception:
            if attempt == 0:
                logger.warning(
                    "Raw SVG upload failed for %s/%s; retrying once",
                    raw_svg.domain,
                    raw_svg.source_id,
                    exc_info=True,
                )
            else:
                logger.exception("Raw SVG upload failed for %s/%s", raw_svg.domain, raw_svg.source_id)
    return False


def _write_raw_svg_locally(raw_svg: RawSVG, output_dir: str) -> None:
    dest = Path(output_dir) / "raw" / raw_svg.domain / f"{raw_svg.source_id}.svg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(raw_svg.svg_string)
    logger.info("Wrote %s", dest)


def _write_ir_locally(raw_svg: RawSVG, output_dir: str) -> None:
    diagram_ir = raw_svg.metadata.get("diagram_ir")
    if not diagram_ir:
        return
    dest = Path(output_dir) / "ir" / raw_svg.domain / f"{raw_svg.source_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(diagram_ir, indent=2, sort_keys=True))
    logger.info("Wrote %s", dest)


def _update_raster_skipped(raw_svg: RawSVG, stats: RunStats) -> None:
    raster_skip_count = raw_svg.metadata.get("raster_skip_count")
    if isinstance(raster_skip_count, int):
        stats.raster_skipped = max(stats.raster_skipped, raster_skip_count)


def _scrape_valid_svgs(
    scrapers: Iterable[BaseScraper],
    scrape_config: ScrapeConfig,
    pipeline_config: PipelineConfig,
    *,
    dry_run: bool,
    output_dir: str,
    s3_client,
    stats: RunStats,
    require_diagram_ir: bool,
) -> list[RawSVG]:
    valid_svgs: list[RawSVG] = []
    lock = threading.Lock()
    quality_counts: Counter[str] = Counter()
    quality_bucket_size = max(1, int(scrape_config.quality_report_every or 1000))

    def _log_quality_report(checked: int) -> None:
        if not quality_counts:
            logger.info("Quality report after %d SVGs: no warnings", checked)
            return
        parts = ", ".join(f"{key}={value}" for key, value in quality_counts.most_common())
        logger.info("Quality report after %d SVGs: %s", checked, parts)

    def _scrape_one(scraper: BaseScraper) -> list[RawSVG]:
        source_name = scraper.__class__.__name__
        source_valid: list[RawSVG] = []
        try:
            iterator = scraper.scrape(scrape_config)
            for raw_svg in iterator:
                with lock:
                    stats.scraped += 1
                    _update_raster_skipped(raw_svg, stats)
                    should_report = stats.scraped % quality_bucket_size == 0

                ok, reason = validate_svg(raw_svg.svg_string, scrape_config)
                if not ok:
                    with lock:
                        stats.failed_validation += 1
                    logger.info("Validation failed for %s/%s: %s", raw_svg.domain, raw_svg.source_id, reason)
                    continue

                ir_result = label_raw_svg_ir(raw_svg)
                if not ir_result.accepted:
                    logger.info(
                        "IR labeling failed for %s/%s: %s",
                        raw_svg.domain,
                        raw_svg.source_id,
                        ir_result.reason,
                    )
                    if require_diagram_ir:
                        with lock:
                            stats.failed_ir_validation += 1
                        continue

                result: SVGValidationResult = validate_svg_detailed(raw_svg.svg_string, scrape_config)
                score = svg_quality_score(result)
                raw_svg.metadata = dict(raw_svg.metadata)
                raw_svg.metadata["quality_score"] = score
                if result.warnings:
                    raw_svg.metadata["quality_warnings"] = list(result.warnings)
                    with lock:
                        quality_counts.update(result.warnings)
                if should_report:
                    with lock:
                        _log_quality_report(stats.scraped)

                if dry_run:
                    _write_raw_svg_locally(raw_svg, output_dir)
                    if getattr(scrape_config, "dump_ir", False):
                        _write_ir_locally(raw_svg, output_dir)
                elif _upload_raw_svg(raw_svg, pipeline_config, s3_client):
                    with lock:
                        stats.written_to_s3 += 1
                else:
                    continue

                source_valid.append(raw_svg)
        except Exception:
            logger.exception("Unhandled scraper failure in %s; skipping source", source_name)
        return source_valid

    scrapers = list(scrapers)
    max_workers = max(1, min(scrape_config.retrieval_parallelism, len(scrapers) or 1))
    if max_workers == 1:
        for scraper in scrapers:
            valid_svgs.extend(_scrape_one(scraper))
        valid_svgs.sort(key=lambda raw: (-int(raw.metadata.get("quality_score", 0)), raw.domain, raw.source_id))
        return valid_svgs

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_scrape_one, scraper) for scraper in scrapers]
        for future in as_completed(futures):
            valid_svgs.extend(future.result())

    valid_svgs.sort(key=lambda raw: (-int(raw.metadata.get("quality_score", 0)), raw.domain, raw.source_id))
    return valid_svgs


def _write_final_summary(stats: RunStats) -> None:
    rows = [
        ("Scraped", str(stats.scraped)),
        ("Raster-skipped", str(stats.raster_skipped)),
        ("Failed validation", str(stats.failed_validation)),
        ("Failed caption", str(stats.failed_caption)),
        ("Written to S3", str(stats.written_to_s3)),
        ("Manifest URI", stats.manifest_uri or ""),
    ]
    if stats.failed_ir_validation:
        rows.insert(3, ("Failed IR validation", str(stats.failed_ir_validation)))

    stage_width = max(len("Stage"), *(len(stage) for stage, _ in rows))
    count_width = max(len("Count"), *(len(count) for _, count in rows))
    top = f"┌{'─' * (stage_width + 2)}┬{'─' * (count_width + 2)}┐"
    header = f"│ {'Stage'.ljust(stage_width)} │ {'Count'.rjust(count_width)} │"
    sep = f"├{'─' * (stage_width + 2)}┼{'─' * (count_width + 2)}┤"
    body = [f"│ {stage.ljust(stage_width)} │ {count.rjust(count_width)} │" for stage, count in rows]
    bottom = f"└{'─' * (stage_width + 2)}┴{'─' * (count_width + 2)}┘"
    table = "\n".join([top, header, sep, *body, bottom])
    print(table)
    logger.info("\n%s", table)


def run(args: argparse.Namespace) -> RunStats:
    config = _load_pipeline_config(args)
    scrape_config = ScrapeConfig(
        max_svgs_per_source=args.max_per_source,
        retrieval_parallelism=getattr(args, "retrieval_parallelism", 2),
        caption_parallelism=getattr(args, "caption_parallelism", 4),
        caption_backend=getattr(args, "caption_backend", "anthropic"),
        caption_local_base_url=getattr(args, "caption_local_base_url", "http://localhost:11434/v1"),
        caption_local_model=getattr(args, "caption_local_model", "llava"),
        wikimedia_dump_path=getattr(args, "wikimedia_dump_path", ""),
        dump_ir=getattr(args, "dump_ir", False),
        wikimedia_asset_base_url=getattr(
            args,
            "wikimedia_asset_base_url",
            "https://upload.wikimedia.org/wikipedia/commons",
        ),
    )
    stats = RunStats()

    s3_client = None
    if not args.dry_run:
        boto3.setup_default_session(profile_name=config.aws_profile, region_name=config.region)
        s3_client = boto3.Session(profile_name=config.aws_profile, region_name=config.region).client("s3")

    valid_svgs = _scrape_valid_svgs(
        _selected_scrapers(args.sources, getattr(args, "wikipedia_source", "api")),
        scrape_config,
        config,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        s3_client=s3_client,
        stats=stats,
        require_diagram_ir=getattr(args, "require_diagram_ir", False),
    )

    if args.no_caption:
        stats.manifest_uri = "not written (--no-caption)"
        _write_final_summary(stats)
        return stats

    records = []
    if valid_svgs:
        records, caption_failures = batch_caption(valid_svgs, config)
        stats.failed_caption = len(caption_failures)
        for raw_svg, reason in caption_failures:
            logger.info("Caption failed for %s/%s: %s", raw_svg.domain, raw_svg.source_id, reason)

    if records:
        stats.manifest_uri = (
            dry_run_write(records, args.output_dir)
            if args.dry_run
            else write_manifest(records, config)
        )
    else:
        stats.manifest_uri = "not written (no captioned records)"
        logger.warning("No captioned records were produced; manifest not written")

    _write_final_summary(stats)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    import sys

    runner = sys.modules[__name__]
    runner.main()
