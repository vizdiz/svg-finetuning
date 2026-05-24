from __future__ import annotations

from pathlib import Path

from backend.dataset_pipeline.config import ScrapeConfig
from backend.dataset_pipeline.corpus.bulk_sources import (
    arxiv_bulk_candidates,
    arxiv_ids_from_local_paths,
    make_line_aligned_shards,
    merge_ranked_wikimedia_candidates,
    rank_wikimedia_dump_shard,
    rank_wikimedia_dump_candidates,
    write_ranked_wikimedia_candidates,
    read_ranked_wikimedia_candidates,
    wikimedia_candidates_from_dump,
    wikimedia_candidates_from_ranked,
)
from backend.dataset_pipeline.corpus.schema import read_candidates


def _image_insert(rows: list[str]) -> str:
    return "INSERT INTO `image` VALUES " + ",".join(rows) + ";\n"


def _row(filename: str, size: int, width: int, height: int, sha1: str = "abc") -> str:
    values = [
        f"'{filename}'",
        str(size),
        str(width),
        str(height),
        "0",
        "0",
        "'DRAWING'",
        "'image'",
        "'svg+xml'",
        "0",
        "0",
        "'20260101000000'",
        f"'{sha1}'",
    ]
    values.extend(["0"] * 10)
    return "(" + ",".join(values) + ")"


def test_wikimedia_dump_candidate_builder_ranks_without_fetching(tmp_path):
    dump = tmp_path / "image.sql"
    dump.write_text(
        _image_insert(
            [
                _row("Logo.svg", 9000, 600, 600, "logo"),
                _row("Detailed_network_diagram.svg", 20000, 1200, 800, "network"),
                _row("System_pipeline_flow_chart.svg", 18000, 1000, 700, "pipeline"),
            ]
        )
    )
    config = ScrapeConfig(max_svgs_per_source=2, min_svg_bytes=500, max_svg_bytes=100000)

    ranked = rank_wikimedia_dump_candidates(dump_path=str(dump), limit=2, config=config)
    result = wikimedia_candidates_from_dump(
        corpus_id="unit_wikimedia",
        dump_path=str(dump),
        output_root=tmp_path,
        limit=2,
        config=config,
    )

    candidates = read_candidates(Path(result.candidates))
    assert [candidate.filename for candidate, _ in ranked] == [
        "System_pipeline_flow_chart.svg",
        "Detailed_network_diagram.svg",
    ]
    assert result.record_count == 2
    assert {candidate.source for candidate in candidates} == {"wikimedia"}
    assert all(candidate.fetch_status == "pending" for candidate in candidates)
    assert all(candidate.provenance["fetch_policy"] == "selected_original_only_after_ranking" for candidate in candidates)


def test_arxiv_bulk_candidates_dedupe_and_preserve_bulk_policy(tmp_path):
    result = arxiv_bulk_candidates(
        corpus_id="unit_arxiv",
        arxiv_ids=["2605.18754v1", "https://arxiv.org/abs/2605.18754", "2501.01234v2"],
        output_root=tmp_path,
    )

    candidates = read_candidates(Path(result.candidates))
    assert result.record_count == 3
    assert {candidate.source for candidate in candidates} == {"arxiv"}
    assert candidates[0].uri == "arxiv://source/2605.18754v1"
    assert candidates[0].provenance["fetch_policy"] == "s3_source_tarball_only"


def test_arxiv_ids_from_local_paths():
    ids = arxiv_ids_from_local_paths(
        [
            Path("scratch/svgs/arxiv/2605.18754v1_0.svg"),
            Path("scratch/svgs/arxiv/not-an-id.svg"),
            Path("scratch/svgs/arxiv/2501.01234v2_3.svg"),
        ]
    )

    assert ids == ["2605.18754v1", "2501.01234v2"]


def test_wikimedia_dump_shards_rank_and_reduce(tmp_path):
    dump = tmp_path / "image.sql"
    dump.write_text(
        _image_insert([_row("Network_diagram_a.svg", 20000, 1200, 800, "a")])
        + _image_insert([_row("Pipeline_flow_chart_b.svg", 22000, 1400, 900, "b")])
        + _image_insert([_row("Tiny.svg", 100, 10, 10, "tiny")])
    )
    config = ScrapeConfig(max_svgs_per_source=2, min_svg_bytes=500, max_svg_bytes=100000)

    shards = make_line_aligned_shards(dump, 2)
    ranked_files = []
    for shard in shards:
        ranked = rank_wikimedia_dump_shard(
            dump_path=dump,
            start=shard["start"],
            end=shard["end"],
            limit=2,
            config=config,
        )
        output = tmp_path / f"ranked_{shard['shard_index']}.jsonl"
        write_ranked_wikimedia_candidates(output, ranked)
        ranked_files.append(output)

    merged = merge_ranked_wikimedia_candidates(read_ranked_wikimedia_candidates(ranked_files), limit=2)
    result = wikimedia_candidates_from_ranked(
        corpus_id="unit_wikimedia_distributed",
        ranked=merged,
        output_root=tmp_path,
        dump_path=str(dump),
        asset_base_url="https://upload.wikimedia.org/wikipedia/commons",
    )

    candidates = read_candidates(Path(result.candidates))
    assert len(candidates) == 2
    assert {candidate.metadata["filename"] for candidate in candidates} == {
        "Network_diagram_a.svg",
        "Pipeline_flow_chart_b.svg",
    }
