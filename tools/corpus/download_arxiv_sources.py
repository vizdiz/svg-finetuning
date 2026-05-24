from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.dataset_pipeline.corpus.s3_orchestration import download_arxiv_source_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-candidates", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-request-payer", action="store_true")
    args = parser.parse_args()
    result = download_arxiv_source_batch(
        input_candidates=Path(args.input_candidates),
        destination_root=Path(args.destination_root),
        limit=args.limit or None,
        request_payer=not args.no_request_payer,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
