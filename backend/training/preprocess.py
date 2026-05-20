"""
training.preprocess

Stub preprocessing entrypoint for a SageMaker Processing step.

The real work is intentionally lightweight:
  - validate the dataset manifest
  - summarize the training corpus
  - write a JSON summary for the next step
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PreprocessSummary:
    manifest_path: str
    record_count: int
    files: list[str]
    split_counts: dict[str, int]


def build_summary(manifest: dict, manifest_path: str) -> PreprocessSummary:
    files = list(manifest.get("files", []))
    split_counts = {
        "train": int(manifest.get("num_train", 0)),
        "val": int(manifest.get("num_val", 0)),
    }
    return PreprocessSummary(
        manifest_path=manifest_path,
        record_count=int(manifest.get("record_count", manifest.get("total", 0))),
        files=files,
        split_counts=split_counts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess dataset manifest")
    parser.add_argument("--manifest", required=True, help="Path to a dataset_manifest.json file")
    parser.add_argument("--output-dir", required=True, help="Directory for processing outputs")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    summary = build_summary(manifest, str(manifest_path))

    output_path = output_dir / "preprocess_summary.json"
    output_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True))
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

