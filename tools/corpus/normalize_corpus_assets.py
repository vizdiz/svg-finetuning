from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.dataset_pipeline.corpus.normalize_assets import normalize_fetched_assets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-candidates", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--output-root", default="pipeline_output/corpus")
    parser.add_argument("--require-ir", action="store_true")
    args = parser.parse_args()
    result = normalize_fetched_assets(
        corpus_id=args.corpus_id,
        input_candidates=Path(args.input_candidates),
        output_root=Path(args.output_root),
        require_ir=args.require_ir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
