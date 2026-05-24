from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.dataset_pipeline.corpus.assemble import assemble_dataset_products


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--records", action="append", required=True)
    parser.add_argument("--output-root", default="pipeline_output/datasets")
    args = parser.parse_args()
    result = assemble_dataset_products(
        dataset_id=args.dataset_id,
        normalized_record_files=[Path(path) for path in args.records],
        output_root=Path(args.output_root),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
