from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.dataset_pipeline.corpus.synthetic_strict_ir import write_synthetic_corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-id", default="")
    parser.add_argument("--output-root", default="pipeline_output/corpus")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--upload-bucket", default="")
    args = parser.parse_args()

    corpus_id = args.corpus_id or f"synthetic_strict_ir_{args.count}_seed{args.seed}"
    result = write_synthetic_corpus(
        corpus_id=corpus_id,
        output_root=Path(args.output_root),
        count=args.count,
        seed=args.seed,
        upload_bucket=args.upload_bucket,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
