from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.dataset_pipeline.corpus.eval_gate import build_eval_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_eval_report(records_path=Path(args.records), output_path=Path(args.output))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
