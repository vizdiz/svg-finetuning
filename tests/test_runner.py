"""
Unit tests for dataset_pipeline.pipeline.runner.

Run with:
    python -m pytest tests/runner.py -v
"""

import json
import threading
import time
from types import SimpleNamespace

from backend.dataset_pipeline.pipeline import runner
from backend.dataset_pipeline.processing.captioner import TrainingRecord
from backend.training.diagram_compiler import compile_diagram_ir
from backend.training.diagram_ir import (
    CanvasSpec,
    DiagramIRDocument,
    EdgeSpec,
    LayoutSpec,
    NodeSpec,
    PortSpec,
)
from backend.dataset_pipeline.scrapers.base_scraper import RawSVG


def _raw(source_id: str, domain: str = "arxiv", svg: str = "<svg/>", **metadata) -> RawSVG:
    return RawSVG(
        svg_string=svg,
        source_url=f"https://example.com/{source_id}",
        source_id=source_id,
        domain=domain,
        metadata=metadata,
    )


def _record(raw_svg: RawSVG) -> TrainingRecord:
    return TrainingRecord(
        id=raw_svg.source_id,
        prompt=f"caption {raw_svg.source_id}",
        completion=raw_svg.svg_string,
        source=raw_svg.domain,
        split="",
        metadata=raw_svg.metadata,
    )


def _diagram_svg() -> str:
    doc = DiagramIRDocument(
        diagram_type="flowchart",
        title="Flow",
        canvas=CanvasSpec(width=800, height=400),
        nodes=[
            NodeSpec(id="a", kind="box", label="A", ports=[PortSpec(name="out", side="right")]),
            NodeSpec(id="b", kind="box", label="B", ports=[PortSpec(name="in", side="left")]),
        ],
        edges=[EdgeSpec(id="a-b", source="a", target="b", source_port="out", target_port="in")],
        layout=LayoutSpec(direction="horizontal", alignment="center", spacing=48),
    )
    return compile_diagram_ir(doc)


class _FakeScraper:
    def __init__(self, records=None, exc=None):
        self.records = records or []
        self.exc = exc

    def scrape(self, config):
        if self.exc:
            raise self.exc
        yield from self.records


class _FakeS3:
    def __init__(self, failures_by_key=None):
        self.failures_by_key = dict(failures_by_key or {})
        self.puts = []

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        remaining = self.failures_by_key.get(key, 0)
        if remaining:
            self.failures_by_key[key] = remaining - 1
            raise RuntimeError(f"upload failed for {key}")
        self.puts.append(kwargs)


def _patch_config(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_load_pipeline_config",
        lambda args: SimpleNamespace(
            aws_profile="profile",
            region="us-east-1",
            s3_data_bucket="bucket",
            raw_prefix="raw/",
            anthropic_api_key="key",
        ),
    )


def test_dry_run_validates_captions_and_writes_local_manifest(monkeypatch, tmp_path):
    _patch_config(monkeypatch)
    good = _raw("good", "arxiv", "<svg>good</svg>", raster_skip_count=3)
    bad = _raw("bad", "arxiv", "<svg>bad</svg>")
    monkeypatch.setattr(runner, "_selected_scrapers", lambda source, wikipedia_source="api": [_FakeScraper([good, bad])])
    monkeypatch.setattr(runner, "validate_svg", lambda svg, config: (svg != "<svg>bad</svg>", "bad_svg"))

    captured = {}

    def fake_batch_caption(raw_svgs, config):
        captured["captioned"] = raw_svgs
        return [_record(raw_svgs[0])], []

    def fake_dry_run_write(records, output_dir):
        captured["records"] = records
        captured["output_dir"] = output_dir
        return f"{output_dir}/manifest.json"

    monkeypatch.setattr(runner, "batch_caption", fake_batch_caption)
    monkeypatch.setattr(runner, "dry_run_write", fake_dry_run_write)

    stats = runner.run(
        SimpleNamespace(sources="arxiv", max_per_source=5, dry_run=True, output_dir=str(tmp_path), no_caption=False)
    )

    assert stats.scraped == 2
    assert stats.raster_skipped == 3
    assert stats.failed_validation == 1
    assert stats.failed_caption == 0
    assert stats.written_to_s3 == 0
    assert stats.manifest_uri == f"{tmp_path}/manifest.json"
    assert captured["captioned"] == [good]
    assert captured["records"] == [_record(good)]
    assert (tmp_path / "raw" / "arxiv" / "good.svg").read_text() == "<svg>good</svg>"
    assert not (tmp_path / "raw" / "arxiv" / "bad.svg").exists()


def test_non_dry_run_retries_raw_upload_and_skips_failed_upload(monkeypatch):
    _patch_config(monkeypatch)
    retry_then_ok = _raw("retry-then-ok")
    always_fails = _raw("always-fails")
    fake_s3 = _FakeS3(
        failures_by_key={
            "raw/arxiv/retry-then-ok.svg": 1,
            "raw/arxiv/always-fails.svg": 2,
        }
    )
    monkeypatch.setattr(runner.boto3, "setup_default_session", lambda **kwargs: None)
    monkeypatch.setattr(
        runner.boto3,
        "Session",
        lambda **kwargs: SimpleNamespace(client=lambda name: fake_s3),
    )
    monkeypatch.setattr(
        runner,
        "_selected_scrapers",
        lambda source, wikipedia_source="api": [_FakeScraper([retry_then_ok, always_fails])],
    )
    monkeypatch.setattr(runner, "validate_svg", lambda svg, config: (True, "ok"))
    monkeypatch.setattr(runner, "batch_caption", lambda raw_svgs, config: ([_record(raw_svgs[0])], []))
    monkeypatch.setattr(runner, "write_manifest", lambda records, config: "s3://bucket/train/dataset_manifest.json")

    stats = runner.run(
        SimpleNamespace(sources="arxiv", max_per_source=5, dry_run=False, output_dir="unused", no_caption=False)
    )

    assert [put["Key"] for put in fake_s3.puts] == ["raw/arxiv/retry-then-ok.svg"]
    assert stats.scraped == 2
    assert stats.written_to_s3 == 1
    assert stats.manifest_uri == "s3://bucket/train/dataset_manifest.json"


def test_caption_failures_are_logged_and_skipped(monkeypatch, tmp_path):
    _patch_config(monkeypatch)
    ok = _raw("ok")
    failed = _raw("failed")
    monkeypatch.setattr(runner, "_selected_scrapers", lambda source, wikipedia_source="api": [_FakeScraper([ok, failed])])
    monkeypatch.setattr(runner, "validate_svg", lambda svg, config: (True, "ok"))
    monkeypatch.setattr(runner, "batch_caption", lambda raw_svgs, config: ([_record(ok)], [(failed, "caption_failed")]))
    monkeypatch.setattr(runner, "dry_run_write", lambda records, output_dir: f"{output_dir}/manifest.json")

    stats = runner.run(
        SimpleNamespace(sources="arxiv", max_per_source=5, dry_run=True, output_dir=str(tmp_path), no_caption=False)
    )

    assert stats.scraped == 2
    assert stats.failed_caption == 1
    assert stats.manifest_uri == f"{tmp_path}/manifest.json"


def test_scraper_exception_skips_source_and_continues(monkeypatch, tmp_path):
    _patch_config(monkeypatch)
    good = _raw("wiki", "wikipedia")
    monkeypatch.setattr(
        runner,
        "_selected_scrapers",
        lambda source, wikipedia_source="api": [_FakeScraper(exc=RuntimeError("boom")), _FakeScraper([good])],
    )
    monkeypatch.setattr(runner, "validate_svg", lambda svg, config: (True, "ok"))
    monkeypatch.setattr(runner, "batch_caption", lambda raw_svgs, config: ([_record(good)], []))
    monkeypatch.setattr(runner, "dry_run_write", lambda records, output_dir: f"{output_dir}/manifest.json")

    stats = runner.run(
        SimpleNamespace(sources="both", max_per_source=5, dry_run=True, output_dir=str(tmp_path), no_caption=False)
    )

    assert stats.scraped == 1
    assert stats.failed_validation == 0
    assert stats.manifest_uri == f"{tmp_path}/manifest.json"


def test_no_caption_outputs_raw_only_and_suppresses_manifest(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch)
    good = _raw("raw-only", "wikipedia")
    monkeypatch.setattr(runner, "_selected_scrapers", lambda source, wikipedia_source="api": [_FakeScraper([good])])
    monkeypatch.setattr(runner, "validate_svg", lambda svg, config: (True, "ok"))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("caption or manifest path should not run")

    monkeypatch.setattr(runner, "batch_caption", fail_if_called)
    monkeypatch.setattr(runner, "dry_run_write", fail_if_called)
    monkeypatch.setattr(runner, "write_manifest", fail_if_called)

    stats = runner.run(
        SimpleNamespace(sources="wikipedia", max_per_source=5, dry_run=True, output_dir=str(tmp_path), no_caption=True)
    )

    assert stats.scraped == 1
    assert stats.manifest_uri == "not written (--no-caption)"
    assert (tmp_path / "raw" / "wikipedia" / "raw-only.svg").exists()
    assert "│ Manifest URI      │ not written (--no-caption) │" in capsys.readouterr().out


def test_dry_run_no_caption_does_not_require_aws_or_anthropic_env(monkeypatch, tmp_path):
    for key in ("AWS_PROFILE", "S3_DATA_BUCKET", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    good = _raw("env-light", "arxiv")
    monkeypatch.setattr(runner, "_selected_scrapers", lambda source, wikipedia_source="api": [_FakeScraper([good])])
    monkeypatch.setattr(runner, "validate_svg", lambda svg, config: (True, "ok"))
    monkeypatch.setattr(runner, "batch_caption", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    stats = runner.run(
        SimpleNamespace(sources="arxiv", max_per_source=5, dry_run=True, output_dir=str(tmp_path), no_caption=True)
    )

    assert stats.scraped == 1
    assert stats.manifest_uri == "not written (--no-caption)"


def test_require_diagram_ir_filters_non_roundtrippable_svgs(monkeypatch, tmp_path):
    _patch_config(monkeypatch)
    good = _raw("good", "arxiv", _diagram_svg())
    bad = _raw("bad", "arxiv", "<svg><rect /></svg>")
    monkeypatch.setattr(runner, "_selected_scrapers", lambda source, wikipedia_source="api": [_FakeScraper([good, bad])])
    monkeypatch.setattr(runner, "validate_svg", lambda svg, config: (True, "ok"))

    captured = {}

    def fake_batch_caption(raw_svgs, config):
        captured["captioned"] = raw_svgs
        return [_record(raw_svgs[0])], []

    monkeypatch.setattr(runner, "batch_caption", fake_batch_caption)
    monkeypatch.setattr(runner, "dry_run_write", lambda records, output_dir: f"{output_dir}/manifest.json")

    stats = runner.run(
        SimpleNamespace(
            sources="arxiv",
            max_per_source=5,
            dry_run=True,
            output_dir=str(tmp_path),
            no_caption=False,
            require_diagram_ir=True,
        )
    )

    assert stats.scraped == 2
    assert stats.failed_ir_validation == 1
    assert captured["captioned"] == [good]
    assert captured["captioned"][0].metadata["diagram_ir_accepted"] is True


def test_dump_ir_writes_ir_json_for_accepted_records(monkeypatch, tmp_path):
    _patch_config(monkeypatch)
    good = _raw("dump-ir", "wikipedia", _diagram_svg())
    monkeypatch.setattr(runner, "_selected_scrapers", lambda source, wikipedia_source="api": [_FakeScraper([good])])
    monkeypatch.setattr(runner, "validate_svg", lambda svg, config: (True, "ok"))

    stats = runner.run(
        SimpleNamespace(
            sources="wikipedia",
            max_per_source=5,
            dry_run=True,
            output_dir=str(tmp_path),
            no_caption=True,
            dump_ir=True,
        )
    )

    ir_path = tmp_path / "ir" / "wikipedia" / "dump-ir.json"
    assert stats.scraped == 1
    assert ir_path.exists()
    ir = json.loads(ir_path.read_text())
    assert ir["diagram_type"] == "architecture"


def test_retrieval_parallelism_runs_selected_sources_concurrently(monkeypatch, tmp_path):
    _patch_config(monkeypatch)
    first_started = threading.Event()
    second_started = threading.Event()

    class BlockingScraper:
        def __init__(self, source_id, started, other_started):
            self.record = _raw(source_id, source_id)
            self.started = started
            self.other_started = other_started

        def scrape(self, config):
            self.started.set()
            assert self.other_started.wait(1), "sources did not overlap"
            time.sleep(0.01)
            yield self.record

    monkeypatch.setattr(
        runner,
        "_selected_scrapers",
        lambda source, wikipedia_source="api": [
            BlockingScraper("first", first_started, second_started),
            BlockingScraper("second", second_started, first_started),
        ],
    )
    monkeypatch.setattr(runner, "validate_svg", lambda svg, config: (True, "ok"))

    stats = runner.run(
        SimpleNamespace(
            sources="both",
            max_per_source=5,
            retrieval_parallelism=2,
            dry_run=True,
            output_dir=str(tmp_path),
            no_caption=True,
        )
    )

    assert stats.scraped == 2
    assert (tmp_path / "raw" / "first" / "first.svg").exists()
    assert (tmp_path / "raw" / "second" / "second.svg").exists()
