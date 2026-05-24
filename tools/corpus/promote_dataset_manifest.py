from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.dataset_pipeline.corpus.promotion import promote_dataset_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-manifest", required=True)
    parser.add_argument("--eval-report", required=True)
    parser.add_argument("--data-bucket", required=True)
    parser.add_argument("--s3-dataset-prefix", required=True)
    parser.add_argument("--approved", action="store_true")
    args = parser.parse_args()
    result = promote_dataset_manifest(
        product_manifest_path=Path(args.product_manifest),
        eval_report_path=Path(args.eval_report),
        data_bucket=args.data_bucket,
        s3_dataset_prefix=args.s3_dataset_prefix,
        approved=args.approved,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
