from __future__ import annotations

from io import BytesIO
import gzip
import json
from pathlib import Path

from backend.dataset_pipeline.corpus.caption_queue import build_model_caption_queue
from backend.dataset_pipeline.corpus.commoncrawl_index import fetch_commoncrawl_index_manifest
from backend.dataset_pipeline.corpus.indexed_fetch import fetch_indexed_svg_candidates
from backend.dataset_pipeline.corpus.promotion import promote_dataset_manifest
from backend.dataset_pipeline.corpus.render_artifacts import build_render_artifacts
from backend.dataset_pipeline.corpus.s3_orchestration import download_arxiv_source_batch, read_commoncrawl_warc_record
from backend.dataset_pipeline.corpus.schema import CorpusCandidate, write_jsonl


VALID_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 80"><rect width="40" height="20"/><circle cx="60" cy="40" r="10"/><path d="M10 10 L30 30"/><text x="4" y="70">API</text><line x1="1" y1="1" x2="2" y2="2"/></svg>'


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.downloads: list[dict] = []
        self.puts: list[dict] = []
        self.gets: list[dict] = []

    def download_file(self, bucket, key, filename, ExtraArgs=None):
        self.downloads.append({"bucket": bucket, "key": key, "filename": filename, "ExtraArgs": ExtraArgs})
        Path(filename).write_bytes(self.objects[(bucket, key)])

    def get_object(self, **kwargs):
        self.gets.append(kwargs)
        body = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        range_header = kwargs.get("Range")
        if range_header:
            start, end = range_header.removeprefix("bytes=").split("-", 1)
            body = body[int(start) : int(end) + 1]
        return {"Body": _Body(body)}

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        body = kwargs["Body"]
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = body if isinstance(body, bytes) else body.encode()

    def list_objects_v2(self, **kwargs):
        prefix = kwargs.get("Prefix", "")
        contents = [{"Key": key} for bucket, key in self.objects if bucket == kwargs["Bucket"] and key.startswith(prefix)]
        return {"Contents": contents}


def test_arxiv_s3_batch_downloads_requester_pays_sources(tmp_path):
    candidates_path = tmp_path / "candidates.jsonl"
    candidate = CorpusCandidate(
        candidate_id="arxiv-one",
        source="arxiv",
        uri="s3://arxiv/src/2605.00001.tar",
        target_format="raw_svg",
        metadata={"arxiv_id": "2605.00001"},
    )
    write_jsonl(candidates_path, [candidate.to_dict()])
    s3 = FakeS3()
    s3.objects[("arxiv", "src/2605.00001.tar")] = b"tar-bytes"

    result = download_arxiv_source_batch(input_candidates=candidates_path, destination_root=tmp_path / "downloads", s3_client=s3)

    assert result["stats"]["downloaded"] == 1
    assert s3.downloads[0]["ExtraArgs"] == {"RequestPayer": "requester"}
    assert (tmp_path / "downloads" / "2605.00001.tar").read_bytes() == b"tar-bytes"


def test_commoncrawl_warc_byte_range_read_and_indexed_fetch(tmp_path):
    wrapped = b"HTTP/1.1 200 OK\r\nContent-Type: image/svg+xml\r\n\r\n" + VALID_SVG.encode()
    payload = gzip.compress(wrapped)
    s3 = FakeS3()
    s3.objects[("commoncrawl", "crawl-data/segment/file.warc.gz")] = b"prefix" + payload + b"suffix"
    candidate = CorpusCandidate(
        candidate_id="cc-one",
        source="commoncrawl",
        uri="https://data.commoncrawl.org/crawl-data/segment/file.warc.gz",
        target_format="raw_svg",
        metadata={"offset": 6, "length": len(payload)},
        priority_score=0.8,
    )
    assert read_commoncrawl_warc_record(candidate, s3_client=s3).startswith("<svg")
    assert s3.gets[0]["Range"] == f"bytes=6-{6 + len(payload) - 1}"

    candidates_path = tmp_path / "cc.jsonl"
    write_jsonl(candidates_path, [candidate.to_dict()])
    fetched = fetch_indexed_svg_candidates(
        corpus_id="cc_fetch",
        input_candidates=candidates_path,
        output_root=tmp_path,
        fetch_fn=lambda row: read_commoncrawl_warc_record(row, s3_client=s3),
    )
    assert fetched["stats"]["fetched"] == 1


def test_commoncrawl_index_fetcher_extracts_svg_records(tmp_path):
    s3 = FakeS3()
    index_key = "cc-index/collections/CC-MAIN-2026-17/indexes/cdx-00000.gz"
    raw = "\n".join(
        [
            'org,example)/a.svg 20260401010101 {"url":"https://example.org/a.svg","mime":"image/svg+xml","status":"200","digest":"d1","length":"111","offset":"22","filename":"crawl-data/CC-MAIN-2026-17/segments/x/warc/test.warc.gz"}',
            'org,example)/b.html 20260401010102 {"url":"https://example.org/b.html","mime":"text/html","status":"200","digest":"d2","length":"222","offset":"33","filename":"crawl-data/CC-MAIN-2026-17/segments/x/warc/test.warc.gz"}',
        ]
    ).encode()
    s3.objects[("commoncrawl", index_key)] = gzip.compress(raw)

    summary = fetch_commoncrawl_index_manifest(
        crawl_id="CC-MAIN-2026-17",
        output_path=tmp_path / "commoncrawl.jsonl",
        s3_client=s3,
        index_keys=[index_key],
        max_files=1,
        limit=10,
    )

    rows = [json.loads(line) for line in (tmp_path / "commoncrawl.jsonl").read_text().splitlines() if line.strip()]
    assert summary["record_count"] == 1
    assert rows[0]["url"].endswith(".svg")
    assert rows[0]["warc_uri"].startswith("s3://commoncrawl/crawl-data/")
    assert rows[0]["score"] > 0.0


def test_render_artifacts_caption_queue_and_manual_promotion(tmp_path):
    records_path = tmp_path / "records.jsonl"
    write_jsonl(
        records_path,
        [
            {"id": "a", "prompt": "Draw a.", "svg": VALID_SVG, "source": "github"},
            {"id": "b", "prompt": "Draw b.", "svg": VALID_SVG, "source": "github"},
        ],
    )
    render_summary = build_render_artifacts(records_path=records_path, output_dir=tmp_path / "render", thumbnail_width=128)
    assert render_summary["rendered"] == 2
    assert render_summary["near_duplicates"] == 1
    assert (tmp_path / "render" / "thumbnails" / "a.png").exists()

    candidates_path = tmp_path / "candidates.jsonl"
    write_jsonl(
        candidates_path,
        [
            CorpusCandidate(
                candidate_id="caption-me",
                source="github",
                uri="https://example.test/a.svg",
                target_format="raw_svg",
                fetch_status="fetched",
                local_path=str(tmp_path / "a.svg"),
                priority_score=0.95,
                caption="API",
            ).to_dict(),
            CorpusCandidate(
                candidate_id="skip-captioned",
                source="github",
                uri="https://example.test/b.svg",
                target_format="raw_svg",
                fetch_status="fetched",
                local_path=str(tmp_path / "b.svg"),
                priority_score=0.95,
                caption="Draw a fully described API workflow diagram.",
            ).to_dict(),
        ],
    )
    queue_summary = build_model_caption_queue(input_candidates=candidates_path, output_path=tmp_path / "caption_queue.jsonl")
    assert queue_summary["stats"]["queued"] == 1

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.write_text('{"prompt":"x","svg":"<svg></svg>"}\n')
    val_path.write_text('{"prompt":"y","svg":"<svg></svg>"}\n')
    product_manifest = tmp_path / "manifest.json"
    product_manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "unit_raw_svg_train",
                "created_at": "2026-05-24T00:00:00Z",
                "record_count": 2,
                "files": [str(train_path), str(val_path)],
                "split": "train",
                "metadata": {"product": "raw_svg_train"},
            }
        )
    )
    eval_report = tmp_path / "eval_report.json"
    eval_report.write_text(json.dumps({"passed": True, "schema_validity_rate": 1.0, "compilability_rate": 1.0, "render_validity_rate": 1.0}))
    s3 = FakeS3()

    promoted = promote_dataset_manifest(
        product_manifest_path=product_manifest,
        eval_report_path=eval_report,
        data_bucket="data-bucket",
        s3_dataset_prefix="s3://data-bucket/train/unit_raw_svg_train",
        s3_client=s3,
        approved=True,
    )

    assert promoted["promoted_manifest_uri"] == "s3://data-bucket/train/dataset_manifest.json"
    assert ("data-bucket", "train/dataset_manifest.json") in s3.objects
    manifest = json.loads(s3.objects[("data-bucket", "train/dataset_manifest.json")])
    assert all(uri.startswith("s3://") for uri in manifest["files"])
