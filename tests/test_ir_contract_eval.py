from __future__ import annotations

import json

from backend.training.ir_contract_eval import score_ir_contract
from tools.training.build_strict_ir_manifest import build_records


def test_score_ir_contract_accepts_exact_strict_ir():
    record = build_records(copies=1)[0]

    score = score_ir_contract(record, json.dumps(record["diagram_ir"]))

    assert score.schema_valid is True
    assert score.compilable is True
    assert score.label_recall == 1.0
    assert score.edge_recall == 1.0
    assert score.flaws == []


def test_score_ir_contract_rejects_graph_dialect():
    record = build_records(copies=1)[0]
    raw = '{"nodes":[{"id":"browser","label":"browser"}],"edges":[]}'

    score = score_ir_contract(record, raw)

    assert score.schema_valid is False
    assert score.compilable is False
    assert score.label_recall == 0.0
    assert score.edge_recall == 0.0
    assert score.flaws[0].startswith("schema_invalid")


def test_score_ir_contract_reports_missing_edges():
    record = build_records(copies=1)[0]
    payload = dict(record["diagram_ir"])
    payload["edges"] = []

    score = score_ir_contract(record, json.dumps(payload))

    assert score.schema_valid is True
    assert score.compilable is True
    assert score.label_recall == 1.0
    assert score.edge_recall == 0.0
    assert "missing_edges" in score.flaws
