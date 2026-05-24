from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-plan", required=True)
    parser.add_argument("--output-dir", default="pipeline_output/corpus/wikimedia_ranked_shards")
    parser.add_argument("--limit", type=int, default=7500)
    parser.add_argument("--parallelism", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    plan = json.loads(Path(args.shard_plan).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    script = root / "scratch" / "build_bulk_candidates.py"

    jobs: list[tuple[int, Path]] = []
    for shard in plan["shards"]:
        shard_index = int(shard["shard_index"])
        output = output_dir / f"shard_{shard_index:03d}.jsonl"
        if output.exists() and output.stat().st_size > 0 and not args.force:
            continue
        jobs.append((shard_index, output))

    print(
        json.dumps(
            {
                "planned_shards": len(plan["shards"]),
                "pending_shards": len(jobs),
                "parallelism": args.parallelism,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not jobs:
        return 0

    started = time.monotonic()
    completed = 0

    def run_job(shard_index: int, output: Path) -> tuple[int, Path, str]:
        cmd = [
            sys.executable,
            str(script),
            "--mode",
            "wikimedia-rank-shard",
            "--shard-plan",
            args.shard_plan,
            "--shard-index",
            str(shard_index),
            "--limit",
            str(args.limit),
            "--ranked-output",
            str(output),
        ]
        proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"shard {shard_index} failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        return shard_index, output, proc.stdout.strip()

    with ThreadPoolExecutor(max_workers=max(1, args.parallelism)) as pool:
        futures = {pool.submit(run_job, shard_index, output): shard_index for shard_index, output in jobs}
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=30)
            if not done:
                elapsed = int(time.monotonic() - started)
                print(json.dumps({"status": "running", "completed": completed, "pending": len(futures), "elapsed_s": elapsed}), flush=True)
                continue
            for future in done:
                shard_index = futures.pop(future)
                shard_index, output, payload = future.result()
                completed += 1
                elapsed = int(time.monotonic() - started)
                print(
                    json.dumps(
                        {
                            "status": "completed",
                            "shard_index": shard_index,
                            "output": str(output),
                            "completed": completed,
                            "pending": len(futures),
                            "elapsed_s": elapsed,
                            "worker": payload,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
