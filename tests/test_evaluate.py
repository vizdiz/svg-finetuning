from __future__ import annotations

import json

from backend.training.diagram_quality import assess_diagram_ir
from backend.training.dataset_interface import TrainingRecord
from backend.training.evaluate import SemanticDiagnostic, build_report, diagnose_record, main
from tests.test_diagram_quality import _document


def test_build_report_preserves_legacy_val_count():
    report = build_report("s3://bucket/model.tar.gz", 3)

    assert report.model_artifacts == "s3://bucket/model.tar.gz"
    assert report.val_records == 3
    assert report.svg_valid_rate == 1.0
    assert report.metrics == {}


def test_evaluate_local_manifest_reports_dataset_metrics(tmp_path):
    svg = assess_diagram_ir(_document()).compiled_svg
    assert svg is not None

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "svg-record",
                        "prompt": "Draw a client API flow",
                        "svg": svg,
                        "source": "unit",
                    }
                ),
                json.dumps(
                    {
                        "id": "ir-record",
                        "prompt": "Draw the same flow as IR",
                        "diagram_ir": _document().to_dict(),
                        "source": "unit",
                    }
                ),
            ]
        )
        + "\n"
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "unit",
                "created_at": "2026-05-22T00:00:00Z",
                "record_count": 2,
                "files": [str(dataset_path)],
            }
        )
    )

    assert (
        main(
            [
                "--model-artifacts",
                "local",
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(tmp_path / "eval"),
            ]
        )
        == 0
    )

    report = json.loads((tmp_path / "eval" / "evaluation.json").read_text())
    diagnostic_rows = [
        json.loads(line)
        for line in (tmp_path / "eval" / "dataset_diagnostics.jsonl").read_text().splitlines()
    ]

    assert report["val_records"] == 2
    assert report["metrics"]["dataset"]["record_count"] == 2
    assert report["metrics"]["dataset"]["prompt_present_rate"] == 1.0
    assert report["metrics"]["dataset"]["target_svg_valid_rate"] == 1.0
    assert report["metrics"]["dataset"]["target_ir_valid_rate"] == 1.0
    assert report["metrics"]["dataset"]["target_any_valid_rate"] == 1.0
    assert report["metrics"]["dataset"]["diagnostics"]["validity"] == 1.0
    assert report["metrics"]["dataset"]["diagnostics"]["quality_score"] > 0.5
    assert [row["id"] for row in diagnostic_rows] == ["svg-record", "ir-record"]
    assert all("quality_score" in row for row in diagnostic_rows)


def test_diagnose_record_reports_source_agnostic_flaws():
    diagnostic = diagnose_record(
        TrainingRecord(
            id="bad",
            prompt="",
            svg="<svg><script>alert(1)</script></svg>",
        )
    )

    assert diagnostic.validity == 0.0
    assert "missing_prompt" in diagnostic.flaws
    assert "svg_no_dimensions" in diagnostic.flaws
    assert "svg_script_tag" in diagnostic.flaws
    assert diagnostic.semantic_alignment is None


def test_evaluate_local_manifest_can_write_semantic_diagnostics(tmp_path, monkeypatch):
    svg = assess_diagram_ir(_document()).compiled_svg
    assert svg is not None

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "semantic-record",
                "prompt": "Draw a labeled client API flow",
                "svg": svg,
                "source": "unit",
            }
        )
        + "\n"
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "semantic-unit",
                "created_at": "2026-05-22T00:00:00Z",
                "record_count": 1,
                "files": [str(dataset_path)],
            }
        )
    )

    def fake_score(record, client=None, model=""):
        assert record.id == "semantic-record"
        return SemanticDiagnostic(
            id=record.id,
            semantic_alignment=0.9,
            layout_fidelity=0.8,
            label_fidelity=0.7,
            connection_fidelity=0.6,
            geometry_fidelity=1.0,
            diagram_usefulness=0.85,
            flaws=["minor_label_issue"],
        )

    monkeypatch.setattr("backend.training.evaluate.score_semantic_alignment", fake_score)

    assert (
        main(
            [
                "--model-artifacts",
                "local",
                "--manifest",
                str(manifest_path),
                "--semantic-sample-size",
                "1",
                "--output-dir",
                str(tmp_path / "eval"),
            ]
        )
        == 0
    )

    report = json.loads((tmp_path / "eval" / "evaluation.json").read_text())
    semantic_rows = [
        json.loads(line)
        for line in (tmp_path / "eval" / "semantic_diagnostics.jsonl").read_text().splitlines()
    ]

    assert report["metrics"]["dataset"]["semantic"]["sample_size"] == 1
    assert report["metrics"]["dataset"]["semantic"]["semantic_alignment"] == 0.9
    assert report["metrics"]["dataset"]["semantic"]["flaw_counts"] == {"minor_label_issue": 1}
    assert semantic_rows[0]["id"] == "semantic-record"
