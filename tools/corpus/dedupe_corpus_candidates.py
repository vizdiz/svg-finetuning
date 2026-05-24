from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.dataset_pipeline.corpus.dedupe import dedupe_candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--input-candidates", action="append", required=True)
    parser.add_argument("--output-root", default="pipeline_output/corpus")
    args = parser.parse_args()
    result = dedupe_candidates(
        corpus_id=args.corpus_id,
        input_candidates=[Path(path) for path in args.input_candidates],
        output_root=Path(args.output_root),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
