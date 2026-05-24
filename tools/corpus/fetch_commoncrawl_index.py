from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.dataset_pipeline.corpus.commoncrawl_index import fetch_commoncrawl_index_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    result = fetch_commoncrawl_index_manifest(
        crawl_id=args.crawl_id,
        output_path=Path(args.output),
        max_files=args.max_files or None,
        limit=args.limit or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
