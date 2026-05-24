from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.dataset_pipeline.corpus.source_plan import build_default_plan, write_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-records", type=int, default=50000)
    parser.add_argument("--plan-id", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    plan = build_default_plan(args.target_records, plan_id=args.plan_id or None)
    if args.output:
        write_plan(Path(args.output), plan)
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
