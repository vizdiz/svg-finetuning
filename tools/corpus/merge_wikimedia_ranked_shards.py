from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.dataset_pipeline.corpus.bulk_sources import (
    merge_ranked_wikimedia_candidates,
    read_ranked_wikimedia_candidates,
    wikimedia_candidates_from_ranked,
    write_ranked_wikimedia_candidates,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranked-dir", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--output-root", default="pipeline_output/corpus")
    parser.add_argument("--dump-path", required=True)
    parser.add_argument("--limit", type=int, default=7500)
    parser.add_argument("--asset-base-url", default="https://upload.wikimedia.org/wikipedia/commons")
    args = parser.parse_args()

    ranked_dir = Path(args.ranked_dir)
    inputs = sorted(ranked_dir.glob("shard_*.jsonl"))
    ranked = read_ranked_wikimedia_candidates(inputs)
    merged = merge_ranked_wikimedia_candidates(ranked, limit=args.limit)
    output_root = Path(args.output_root)
    corpus_dir = output_root / args.corpus_id
    corpus_dir.mkdir(parents=True, exist_ok=True)
    merged_path = corpus_dir / "ranked_merged.jsonl"
    write_ranked_wikimedia_candidates(merged_path, merged)
    result = wikimedia_candidates_from_ranked(
        corpus_id=args.corpus_id,
        ranked=merged,
        output_root=output_root,
        dump_path=args.dump_path,
        asset_base_url=args.asset_base_url,
        metadata={
            "source": "wikimedia",
            "bulk_only": True,
            "dump_path": args.dump_path,
            "selection": "distributed_ranked_sql_dump_candidates",
            "source_shards": len(inputs),
            "shard_local_records": len(ranked),
        },
    )
    print(
        json.dumps(
            {
                **result.to_dict(),
                "ranked_merged": str(merged_path),
                "input_shards": len(inputs),
                "input_records": len(ranked),
                "merged_records": len(merged),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
