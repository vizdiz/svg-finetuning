from __future__ import annotations

import json

from backend.dataset_pipeline.corpus.source_plan import build_default_plan, write_plan


def test_build_default_plan_allocates_exact_target():
    plan = build_default_plan(50000)

    assert sum(item.target_records for item in plan.allocations) == 50000
    assert {item.source for item in plan.allocations} == {
        "synthetic",
        "arxiv",
        "wikimedia",
        "github",
        "commoncrawl",
    }
    assert plan.quality_gates["minimums"]["node_count"] == 4
    assert plan.storage["training_manifest_trigger"] == "manual_only_until_approved"


def test_write_plan_round_trip(tmp_path):
    path = tmp_path / "plan.json"
    plan = build_default_plan(100000, plan_id="unit-plan")

    write_plan(path, plan)
    payload = json.loads(path.read_text())

    assert payload["plan_id"] == "unit-plan"
    assert payload["target_records"] == 100000
    assert sum(item["target_records"] for item in payload["allocations"]) == 100000
