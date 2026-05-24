from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.dataset_pipeline.corpus.render_artifacts import build_render_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--duplicate-hamming-threshold", type=int, default=4)
    parser.add_argument("--thumbnail-width", type=int, default=512)
    args = parser.parse_args()
    result = build_render_artifacts(
        records_path=Path(args.records),
        output_dir=Path(args.output_dir),
        max_records=args.max_records or None,
        duplicate_hamming_threshold=args.duplicate_hamming_threshold,
        thumbnail_width=args.thumbnail_width,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
