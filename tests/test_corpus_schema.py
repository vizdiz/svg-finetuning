from __future__ import annotations

import json

from backend.dataset_pipeline.corpus.schema import CorpusCandidate, build_manifest, read_candidates, stable_id, write_jsonl


def test_corpus_candidate_round_trip(tmp_path):
    candidate = CorpusCandidate(
        candidate_id=stable_id("synthetic", "one"),
        source="synthetic",
        uri="synthetic://one",
        target_format="diagram_ir",
        caption="Draw a diagram",
        license="internal-synthetic",
        priority_score=1.0,
        fetch_status="not_required",
    )
    path = tmp_path / "candidates.jsonl"
    write_jsonl(path, [candidate.to_dict()])

    loaded = read_candidates(path)
    manifest = build_manifest("unit", str(path), loaded)

    assert loaded[0].candidate_id == candidate.candidate_id
    assert manifest.record_count == 1
    assert manifest.source_counts == {"synthetic": 1}
    assert manifest.target_format_counts == {"diagram_ir": 1}


def test_corpus_candidate_rejects_unknown_source():
    try:
        CorpusCandidate(
            candidate_id="bad",
            source="scrape-live-web",
            uri="https://example.com/a.svg",
            target_format="raw_svg",
        ).to_dict()
    except ValueError as exc:
        assert "source must be one of" in str(exc)
    else:
        raise AssertionError("expected invalid source to fail")
