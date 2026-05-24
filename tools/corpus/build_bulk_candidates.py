from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.dataset_pipeline.corpus.bulk_sources import (
    arxiv_bulk_candidates,
    arxiv_ids_from_local_paths,
    commoncrawl_candidates_from_index,
    github_candidates_from_index,
    merge_ranked_wikimedia_candidates,
    rank_wikimedia_dump_shard,
    read_ranked_wikimedia_candidates,
    write_ranked_wikimedia_candidates,
    write_wikimedia_shard_plan,
    wikimedia_candidates_from_dump,
    wikimedia_candidates_from_ranked,
)


def _read_id_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["build", "wikimedia-shard-plan", "wikimedia-rank-shard", "wikimedia-merge-shards"],
        default="build",
    )
    parser.add_argument("--source", choices=["wikimedia", "arxiv", "github", "commoncrawl"], default="")
    parser.add_argument("--corpus-id", default="")
    parser.add_argument("--output-root", default="pipeline_output/corpus")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--wikimedia-dump-path", default="")
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--shard-plan", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--ranked-output", default="")
    parser.add_argument("--ranked-input", action="append", default=[])
    parser.add_argument("--arxiv-id-file", default="")
    parser.add_argument("--arxiv-local-glob", default="")
    parser.add_argument("--index-jsonl", default="")
    args = parser.parse_args()

    if args.mode == "wikimedia-shard-plan":
        if not args.wikimedia_dump_path:
            raise SystemExit("--wikimedia-dump-path is required")
        output = Path(args.shard_plan or Path(args.output_root) / "wikimedia_shards.json")
        plan = write_wikimedia_shard_plan(output, Path(args.wikimedia_dump_path), args.shard_count)
        print(json.dumps({"shard_plan": str(output), "shard_count": len(plan["shards"])}, indent=2, sort_keys=True))
        return 0

    if args.mode == "wikimedia-rank-shard":
        if not args.shard_plan:
            raise SystemExit("--shard-plan is required")
        plan = json.loads(Path(args.shard_plan).read_text())
        shard = plan["shards"][args.shard_index]
        ranked = rank_wikimedia_dump_shard(
            dump_path=Path(shard["path"]),
            start=int(shard["start"]),
            end=int(shard["end"]),
            limit=args.limit,
        )
        output = Path(args.ranked_output or Path(args.output_root) / f"wikimedia_ranked_shard_{args.shard_index}.jsonl")
        write_ranked_wikimedia_candidates(output, ranked)
        print(
            json.dumps(
                {
                    "ranked_output": str(output),
                    "records": len(ranked),
                    "shard_index": args.shard_index,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.mode == "wikimedia-merge-shards":
        if not args.ranked_input:
            raise SystemExit("--ranked-input is required at least once")
        if not args.corpus_id:
            raise SystemExit("--corpus-id is required")
        ranked = merge_ranked_wikimedia_candidates(
            read_ranked_wikimedia_candidates([Path(path) for path in args.ranked_input]),
            limit=args.limit,
        )
        output = Path(args.output_root) / args.corpus_id / "ranked_merged.jsonl"
        write_ranked_wikimedia_candidates(output, ranked)
        result = wikimedia_candidates_from_ranked(
            corpus_id=args.corpus_id,
            ranked=ranked,
            output_root=Path(args.output_root),
            dump_path=args.wikimedia_dump_path or "distributed-shards",
            asset_base_url="https://upload.wikimedia.org/wikipedia/commons",
            metadata={
                "source": "wikimedia",
                "bulk_only": True,
                "dump_path": args.wikimedia_dump_path or "distributed-shards",
                "selection": "distributed_ranked_sql_dump_candidates",
            },
        )
        print(json.dumps({**result.to_dict(), "ranked_merged": str(output), "records": len(ranked)}, indent=2, sort_keys=True))
        return 0

    if not args.source:
        raise SystemExit("--source is required in build mode")
    if not args.corpus_id:
        raise SystemExit("--corpus-id is required in build mode")

    if args.source == "wikimedia":
        if not args.wikimedia_dump_path:
            raise SystemExit("--wikimedia-dump-path is required for --source wikimedia")
        result = wikimedia_candidates_from_dump(
            corpus_id=args.corpus_id,
            dump_path=args.wikimedia_dump_path,
            output_root=Path(args.output_root),
            limit=args.limit,
        )
    elif args.source == "arxiv":
        ids: list[str] = []
        if args.arxiv_id_file:
            ids.extend(_read_id_file(Path(args.arxiv_id_file)))
        if args.arxiv_local_glob:
            ids.extend(arxiv_ids_from_local_paths(Path().glob(args.arxiv_local_glob)))
        if not ids:
            raise SystemExit("--arxiv-id-file or --arxiv-local-glob is required for --source arxiv")
        result = arxiv_bulk_candidates(
            corpus_id=args.corpus_id,
            arxiv_ids=ids,
            output_root=Path(args.output_root),
        )
    elif args.source == "github":
        if not args.index_jsonl:
            raise SystemExit("--index-jsonl is required for --source github")
        result = github_candidates_from_index(
            corpus_id=args.corpus_id,
            index_path=Path(args.index_jsonl),
            output_root=Path(args.output_root),
            limit=args.limit,
        )
    else:
        if not args.index_jsonl:
            raise SystemExit("--index-jsonl is required for --source commoncrawl")
        result = commoncrawl_candidates_from_index(
            corpus_id=args.corpus_id,
            index_path=Path(args.index_jsonl),
            output_root=Path(args.output_root),
            limit=args.limit,
        )

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
