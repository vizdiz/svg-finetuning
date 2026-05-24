from __future__ import annotations

import json
from pathlib import Path

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.corpus.fetch_validate import fetch_validate_candidates
from backend.dataset_pipeline.corpus.schema import CorpusCandidate, read_candidates, write_jsonl


VALID_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 80"><rect width="40" height="20"/><circle cx="60" cy="40" r="10"/><path d="M10 10 L30 30"/><text x="4" y="70">API</text><line x1="1" y1="1" x2="2" y2="2"/></svg>'
INVALID_SVG = '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'


def _candidate(candidate_id: str, uri: str) -> CorpusCandidate:
    return CorpusCandidate(
        candidate_id=candidate_id,
        source="wikimedia",
        uri=uri,
        target_format="diagram_ir",
        title=f"File:{candidate_id}.svg",
        fetch_status="pending",
        priority_score=0.8,
    )


def test_fetch_validate_candidates_writes_assets_and_statuses(tmp_path):
    input_path = tmp_path / "input.jsonl"
    write_jsonl(
        input_path,
        [
            _candidate("ok", "https://example.test/ok.svg").to_dict(),
            _candidate("bad", "https://example.test/bad.svg").to_dict(),
            _candidate("missing", "https://example.test/missing.svg").to_dict(),
        ],
    )

    def fetch(candidate: CorpusCandidate) -> str:
        if candidate.candidate_id == "ok":
            return VALID_SVG
        if candidate.candidate_id == "bad":
            return INVALID_SVG
        raise RuntimeError("not found")

    result = fetch_validate_candidates(
        corpus_id="unit_wikimedia_fetched",
        input_candidates=input_path,
        output_root=tmp_path / "out",
        config=ScrapeConfig(min_svg_bytes=0),
        fetch_fn=fetch,
        asset_s3_prefix="s3://bucket/corpus/unit_wikimedia_fetched/assets/wikimedia",
    )

    candidates = read_candidates(Path(result.candidates))
    by_id = {candidate.candidate_id: candidate for candidate in candidates}

    assert result.stats.fetched == 1
    assert result.stats.rejected == 1
    assert result.stats.failed == 1
    assert by_id["ok"].fetch_status == "fetched"
    assert by_id["ok"].content_sha256
    assert by_id["ok"].s3_uri.startswith("s3://bucket/corpus/unit_wikimedia_fetched/assets/wikimedia/")
    assert Path(by_id["ok"].local_path).read_text() == VALID_SVG
    assert by_id["bad"].fetch_status == "rejected"
    assert "script_tag" in by_id["bad"].metadata["validation"]["hard_reject_reasons"]
    assert by_id["missing"].fetch_status == "failed"

    manifest = json.loads(Path(result.manifest).read_text())
    summary = json.loads(Path(result.summary).read_text())
    assert manifest["record_count"] == 3
    assert summary["stats"]["fetched"] == 1


def test_fetch_validate_candidates_limit_leaves_rest_pending(tmp_path):
    input_path = tmp_path / "input.jsonl"
    write_jsonl(
        input_path,
        [
            _candidate("one", "https://example.test/one.svg").to_dict(),
            _candidate("two", "https://example.test/two.svg").to_dict(),
        ],
    )

    result = fetch_validate_candidates(
        corpus_id="unit_limited",
        input_candidates=input_path,
        output_root=tmp_path / "out",
        config=ScrapeConfig(min_svg_bytes=0),
        fetch_fn=lambda _: VALID_SVG,
        limit=1,
    )

    candidates = read_candidates(Path(result.candidates))
    assert [candidate.fetch_status for candidate in candidates] == ["fetched", "pending"]
    assert result.stats.limited is True
