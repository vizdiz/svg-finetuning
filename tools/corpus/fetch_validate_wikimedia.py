from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.corpus.fetch_validate import fetch_validate_candidates


def _run_aws(args: list[str]) -> None:
    subprocess.run(["aws", *args], check=True)


def _is_s3_uri(value: str) -> bool:
    return value.startswith("s3://")


def _prepare_input(value: str, work_dir: Path) -> Path:
    if not _is_s3_uri(value):
        return Path(value)
    local = work_dir / "input_candidates.jsonl"
    _run_aws(["s3", "cp", value, str(local), "--only-show-errors"])
    return local


def _upload_output(output_root: Path, corpus_id: str, s3_prefix: str) -> None:
    source = output_root / corpus_id
    _run_aws(["s3", "sync", str(source), s3_prefix.rstrip("/") + "/", "--only-show-errors"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-candidates", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--output-root", default="pipeline_output/corpus")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--upload-s3-prefix", default="")
    parser.add_argument("--min-svg-bytes", type=int, default=500)
    parser.add_argument("--max-svg-bytes", type=int, default=5_000_000)
    parser.add_argument("--min-svg-elements", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, action="append", default=[1.0, 2.0, 4.0])
    parser.add_argument("--request-delay-s", type=float, default=1.0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="svg-fetch-validate-") as tmp:
        input_candidates = _prepare_input(args.input_candidates, Path(tmp))
        result = fetch_validate_candidates(
            corpus_id=args.corpus_id,
            input_candidates=input_candidates,
            output_root=Path(args.output_root),
            config=ScrapeConfig(
                min_svg_bytes=args.min_svg_bytes,
                max_svg_bytes=args.max_svg_bytes,
                min_svg_elements=args.min_svg_elements,
            ),
            limit=args.limit or None,
            retry_delays=args.retry_delay,
            asset_s3_prefix=(
                args.upload_s3_prefix.rstrip("/") + "/assets/wikimedia"
                if args.upload_s3_prefix
                else ""
            ),
            request_delay_s=args.request_delay_s,
        )

    if args.upload_s3_prefix:
        _upload_output(Path(args.output_root), args.corpus_id, args.upload_s3_prefix)

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
