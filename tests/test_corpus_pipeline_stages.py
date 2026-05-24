from __future__ import annotations

import json
import tarfile
from pathlib import Path

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.corpus.arxiv_worker import extract_arxiv_sources
from backend.dataset_pipeline.corpus.assemble import assemble_dataset_products
from backend.dataset_pipeline.corpus.bulk_sources import (
    arxiv_bulk_candidates,
    commoncrawl_candidates_from_index,
    github_candidates_from_index,
)
from backend.dataset_pipeline.corpus.dedupe import dedupe_candidates
from backend.dataset_pipeline.corpus.eval_gate import build_eval_report
from backend.dataset_pipeline.corpus.indexed_fetch import fetch_indexed_svg_candidates
from backend.dataset_pipeline.corpus.normalize_assets import normalize_fetched_assets
from backend.dataset_pipeline.corpus.schema import CorpusCandidate, read_candidates, write_jsonl
from backend.training.diagram_compiler import compile_diagram_ir
from backend.training.diagram_ir import CanvasSpec, DiagramIRDocument, EdgeSpec, LayoutSpec, NodeSpec


VALID_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 80"><rect width="40" height="20"/><circle cx="60" cy="40" r="10"/><path d="M10 10 L30 30"/><text x="4" y="70">API</text><line x1="1" y1="1" x2="2" y2="2"/></svg>'


def _strict_ir() -> dict:
    doc = DiagramIRDocument(
        diagram_type="flowchart",
        title="Eval",
        canvas=CanvasSpec(width=640, height=480),
        nodes=[NodeSpec(id="a", kind="box", label="A"), NodeSpec(id="b", kind="box", label="B")],
        edges=[EdgeSpec(id="a-b", source="a", target="b")],
        layout=LayoutSpec(direction="horizontal"),
    )
    doc.validate()
    compile_diagram_ir(doc)
    return doc.to_dict()


def test_github_and_commoncrawl_index_manifests(tmp_path):
    github_index = tmp_path / "github.jsonl"
    write_jsonl(
        github_index,
        [
            {"raw_url": "https://raw.githubusercontent.test/repo/a.svg", "repo": "org/repo", "path": "a.svg", "sha": "1", "score": 0.9},
            {"raw_url": "https://raw.githubusercontent.test/repo/b.svg", "repo": "org/repo", "path": "b.svg", "sha": "2", "score": 0.1},
        ],
    )
    cc_index = tmp_path / "cc.jsonl"
    write_jsonl(
        cc_index,
        [
            {"warc_uri": "s3://crawl/segment/file.warc.gz#1", "url": "https://example.test/a.svg", "digest": "d1", "score": 0.7}
        ],
    )

    github = github_candidates_from_index(corpus_id="github_unit", index_path=github_index, output_root=tmp_path, limit=1)
    cc = commoncrawl_candidates_from_index(corpus_id="cc_unit", index_path=cc_index, output_root=tmp_path, limit=5)

    assert read_candidates(Path(github.candidates))[0].source == "github"
    assert read_candidates(Path(github.candidates))[0].uri.endswith("a.svg")
    assert read_candidates(Path(cc.candidates))[0].provenance["fetch_policy"] == "read_warc_record_no_origin_request"


def test_arxiv_source_tarball_extracts_svg_assets(tmp_path):
    seed = arxiv_bulk_candidates(corpus_id="arxiv_seed", arxiv_ids=["2605.18754v1"], output_root=tmp_path)
    source_root = tmp_path / "sources"
    source_root.mkdir()
    tar_path = source_root / "2605.18754v1.tar"
    svg_file = tmp_path / "figure.svg"
    svg_file.write_text(VALID_SVG)
    with tarfile.open(tar_path, "w") as archive:
        archive.add(svg_file, arcname="figures/figure.svg")

    result = extract_arxiv_sources(
        corpus_id="arxiv_extracted",
        input_candidates=Path(seed.candidates),
        source_root=source_root,
        output_root=tmp_path,
        config=ScrapeConfig(min_svg_bytes=0),
    )

    candidates = read_candidates(Path(result["candidates"]))
    assert result["stats"]["assets"] == 1
    assert candidates[0].fetch_status == "fetched"
    assert Path(candidates[0].local_path).exists()


def test_indexed_fetch_dedupe_normalize_assemble_and_eval(tmp_path):
    input_path = tmp_path / "indexed.jsonl"
    candidate = CorpusCandidate(
        candidate_id="gh-one",
        source="github",
        uri="https://example.test/one.svg",
        target_format="raw_svg",
        fetch_status="pending",
        priority_score=0.8,
        caption="Draw a simple API diagram.",
    )
    write_jsonl(input_path, [candidate.to_dict(), candidate.to_dict()])

    fetched = fetch_indexed_svg_candidates(
        corpus_id="indexed_fetched",
        input_candidates=input_path,
        output_root=tmp_path,
        config=ScrapeConfig(min_svg_bytes=0),
        fetch_fn=lambda _: VALID_SVG,
    )
    deduped = dedupe_candidates(
        corpus_id="deduped",
        input_candidates=[Path(fetched["candidates"])],
        output_root=tmp_path,
    )
    normalized = normalize_fetched_assets(
        corpus_id="normalized",
        input_candidates=Path(deduped["candidates"]),
        output_root=tmp_path,
    )
    # Add one strict IR record so assembly/eval can exercise the IR product deterministically.
    records_path = Path(normalized["records"])
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    rows.append({"id": "strict", "prompt": "Draw A to B.", "diagram_ir": _strict_ir(), "svg": "", "source": "synthetic", "metadata": {}})
    write_jsonl(records_path, rows)

    assembled = assemble_dataset_products(
        dataset_id="assembled",
        normalized_record_files=[records_path],
        output_root=tmp_path,
    )
    report = build_eval_report(
        records_path=records_path,
        output_path=tmp_path / "assembled" / "eval_report.json",
        min_render_validity=0.0,
    )

    assert fetched["stats"]["fetched"] == 2
    assert deduped["stats"]["duplicates"] == 1
    assert assembled["products"]["raw_svg_train"]["record_count"] == 1
    assert report["schema_validity_rate"] >= 0.0
