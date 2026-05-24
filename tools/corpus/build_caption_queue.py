from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.dataset_pipeline.corpus.caption_queue import build_model_caption_queue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-priority", type=float, default=0.75)
    parser.add_argument("--min-caption-chars", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    result = build_model_caption_queue(
        input_candidates=Path(args.input_candidates),
        output_path=Path(args.output),
        min_priority=args.min_priority,
        min_caption_chars=args.min_caption_chars,
        limit=args.limit or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
