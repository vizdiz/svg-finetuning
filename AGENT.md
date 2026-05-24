# Agent Context

This file is the handoff context for future agent sessions. Read it before
changing corpus, dataset, training-promotion, or AWS corpus code.

## Corpus Goal

Build a 50K-100K multi-source SVG/diagram corpus. A Wikimedia-only
fetch/validation stage is not the full pipeline.

First target mix:

| Source | Target | Role |
|---|---:|---|
| Synthetic strict IR | 10K-20K | Exact compiler schema and prompt adherence. |
| arXiv bulk source | 20K-40K | Technical diagrams, scientific workflows, complex layouts. |
| Wikimedia metadata + selective SVG fetch | 10K-20K | Clean icon/diagram SVGs with captions/categories. |
| GitHub/Common Crawl indexed SVGs | 10K-20K | Broad SVG diversity, UI/architecture/vector patterns. |

Checked-in 50K default allocation in `backend/dataset_pipeline/corpus/source_plan.py`:

| Source | Target |
|---|---:|
| Synthetic | 10,000 |
| arXiv | 20,000 |
| Wikimedia | 7,500 |
| GitHub | 7,500 |
| Common Crawl | 5,000 |

## Source Policy

Discovery must be bulk/index-only.

- arXiv: requester-pays source tarballs or metadata seeded from arXiv IDs.
- Wikimedia: SQL/XML dumps for file metadata, captions, categories, MIME/type,
  license; fetch only selected ranked SVG originals.
- Common Crawl: index/WAT/WARC records for `.svg` URLs; do not crawl origin
  sites.
- GitHub: BigQuery, GH Archive, or public repo metadata for candidate
  selection; fetch selected raw blobs only after dedupe/ranking.

Live origin fetching is allowed only after candidate ranking and only for the
selected bytes needed by the corpus product.

## End-To-End Pipeline

1. Source discovery from bulk indexes/dumps.
2. Normalized candidate manifest before downloading source bytes.
3. Source-specific bulk fetch/extract.
4. Normalize assets into raw SVG, parsed IR, compiler-canonical IR, prompt or
   caption, provenance/license, and optional render thumbnail.
5. Quality gates and dedupe.
6. Caption and prompt generation, using metadata first and model captioning only
   for high-value records.
7. Dataset product assembly.
8. Training strategy and promotion.
9. Eval gate before deployment.

## Candidate Schema

All sources should write `CorpusCandidate` rows from
`backend/dataset_pipeline/corpus/schema.py`:

```json
{
  "candidate_id": "...",
  "source": "arxiv|wikimedia|github|commoncrawl|synthetic",
  "uri": "s3://... or https://... or source-specific URI",
  "license": "...",
  "title": "...",
  "caption": "...",
  "metadata": {},
  "fetch_status": "not_required|pending|fetched|failed|rejected",
  "priority_score": 0.82
}
```

Candidate manifests are source inventory and state tracking. They are not
training datasets by themselves.

## Dataset Products

Produce multiple datasets, not one mixed blob:

| Product | Contract |
|---|---|
| `strict_ir_train` | prompt -> canonical compiler IR |
| `raw_svg_train` | prompt -> SVG, size-bucketed |
| `svg_to_ir_train` | SVG/caption -> IR |
| `eval_contract` | hand/synthetic prompts with expected nodes/edges |
| `eval_visual` | rendered SVG semantic quality samples |

Do not blindly mix all records. Intended curriculum: synthetic strict IR first,
then real extracted IR, then captioned real diagrams. Raw SVG generation should
remain a separate experiment or late-stage adapter unless context and memory
issues are explicitly solved.

## Quality And Eval Gates

Reject aggressively:

- invalid XML/SVG
- scripts or event handlers
- foreign objects
- embedded raster images
- external references
- missing dimensions/viewBox
- too tiny/empty
- too huge unless bucketed for long-context raw SVG
- duplicate or near-duplicate hashes
- low text/caption alignment
- unparseable or uncompilable IR

Deployment gates should report schema validity, compilability, node label
recall, edge recall, minimum complexity, render validity, canvas overflow or
overlap checks, and semantic alignment sample pass rate.

## Code Map

Implemented:

- `backend/dataset_pipeline/corpus/schema.py`: `CorpusCandidate`,
  `CorpusManifest`, local/S3 prefix helpers.
- `backend/dataset_pipeline/corpus/source_plan.py`: 50K allocation plan.
- `backend/dataset_pipeline/corpus/synthetic_strict_ir.py`: synthetic strict-IR
  generation and training split.
- `backend/dataset_pipeline/corpus/bulk_sources.py`: Wikimedia SQL dump
  ranking/sharding/merge plus arXiv, GitHub, and Common Crawl candidate
  manifest helpers.
- `backend/dataset_pipeline/corpus/fetch_validate.py`: candidate-driven fetch
  and SVG validation stage.
- `backend/dataset_pipeline/corpus/arxiv_worker.py`: arXiv source tarball SVG
  extraction and validation.
- `backend/dataset_pipeline/corpus/indexed_fetch.py`: GitHub raw SVG and
  Common Crawl local-record or remote WARC byte-range fetch/validation.
- `backend/dataset_pipeline/corpus/s3_orchestration.py`: requester-pays arXiv
  source archive batch downloads and Common Crawl WARC byte-range reads with
  adaptive S3 retries.
- `backend/dataset_pipeline/corpus/dedupe.py`: cross-manifest content/URI
  dedupe.
- `backend/dataset_pipeline/corpus/normalize_assets.py`: fetched asset
  normalization into training records with best-effort IR extraction.
- `backend/dataset_pipeline/corpus/assemble.py`: product assembly for
  `strict_ir_train`, `raw_svg_train`, `svg_to_ir_train`, `eval_contract`, and
  `eval_visual`.
- `backend/dataset_pipeline/corpus/eval_gate.py`: schema/compilability/render
  validation report.
- `backend/dataset_pipeline/corpus/render_artifacts.py`: render-thumbnail PNG
  artifacts and near-duplicate render hashes.
- `backend/dataset_pipeline/corpus/caption_queue.py`: model-caption work queue
  for high-value records with missing/short captions.
- `backend/dataset_pipeline/corpus/promotion.py`: explicit manual promotion of
  a passed product manifest to `train/dataset_manifest.json`.
- `backend/dataset_pipeline/corpus/workers.py`: shared source-worker contract.
- `tools/corpus/plan_corpus_build.py`: writes the 50K plan.
- `tools/corpus/build_synthetic_corpus.py`: writes synthetic strict-IR corpus.
- `tools/corpus/build_bulk_candidates.py`: writes Wikimedia/arXiv/GitHub/Common
  Crawl candidate manifests and Wikimedia shard jobs.
- `tools/corpus/run_wikimedia_shards_local.py`: runs Wikimedia rank shards on one
  machine with worker parallelism.
- `tools/corpus/merge_wikimedia_ranked_shards.py`: merges ranked shard outputs.
- `tools/corpus/fetch_validate_wikimedia.py`: fetches selected Wikimedia SVG
  originals from a candidate manifest and writes validation state/assets.
- `tools/corpus/extract_arxiv_sources.py`: extracts SVG assets from arXiv source
  tarballs for an arXiv candidate manifest.
- `tools/corpus/fetch_indexed_sources.py`: fetches selected GitHub/Common Crawl
  indexed SVG candidates.
- `tools/corpus/fetch_commoncrawl_index.py`: fetches Common Crawl CDXJ index
  slices from the public S3 collection and extracts SVG candidates.
- `tools/corpus/download_arxiv_sources.py`: downloads requester-pays arXiv source
  archives from S3 for an arXiv candidate manifest.
- `tools/corpus/dedupe_corpus_candidates.py`: dedupes one or more candidate
  manifests.
- `tools/corpus/normalize_corpus_assets.py`: normalizes fetched assets into training
  records.
- `tools/corpus/assemble_corpus_dataset.py`: assembles dataset products.
- `tools/corpus/build_corpus_eval_report.py`: writes an eval gate report.
- `tools/corpus/render_corpus_thumbnails.py`: writes render thumbnails and render
  hash metadata for visual eval.
- `tools/corpus/build_caption_queue.py`: writes a JSONL model-caption queue.
- `tools/corpus/promote_dataset_manifest.py`: requires `--approved` plus a passed
  eval report before writing the training manifest.

Not implemented:

- BigQuery/GH Archive query jobs; current worker consumes exported JSONL index
  rows.
- A deployed job runner/Step Functions wrapper for the full 50K orchestration;
  current production entry points are explicit CLI commands.

## Current AWS/S3 State

Bucket: `s3://svg-finetuning-data-446224796301/`

Architecture-stage corpus products currently present:

| Prefix | Status |
|---|---|
| `corpus/synthetic_strict_ir_10000_seed101/` | Complete synthetic strict-IR corpus. Manifest says `record_count: 10000`, `source_counts.synthetic: 10000`, `target_format_counts.diagram_ir: 10000`. Training split is 8,947 train / 1,053 val. |
| `corpus/wikimedia_dump_ranked_7500/` | Complete Wikimedia ranking candidate corpus. Manifest says `record_count: 7500`, `source_counts.wikimedia: 7500`, `target_format_counts.diagram_ir: 7500`. Candidate rows are still `fetch_status: pending`. |

Wikimedia ranking artifacts:

- `candidates.jsonl`: 7,500 normalized Wikimedia candidates.
- `manifest.json`
- `ranked_merged.jsonl`
- `ranked_shards/`
- `wikimedia_shards_32.json`

Wikimedia SVG originals are not present for the 7,500 ranked candidates yet.
The ranked candidate corpus has no `assets/` or `raw/` directory. Older dry-run
prefixes contain only small partial fetches and should not be confused with the
7,500-candidate product.

Promoted training data currently present:

| Prefix | Status |
|---|---|
| `train/dataset_manifest.json` | Small 150-record canonical arXiv dataset, not the full 50K corpus. |
| `train/dataset.jsonl`, `val/dataset.jsonl` | Same 150-record promoted training data. |

Older experimental dry-run prefixes:

- `dry-run/aws_ir_dry_run_1k_c7g_arxiv/`
- `dry-run/aws_ir_dry_run_1k_c7g_wikimedia_50k_slice/`
- `dry-run/aws_ir_dry_run_1k_c7g_wikimedia_retry/`
- `dry-run/aws_ir_dry_run_1k_dump/`
- `dry-run/captioned_arxiv_199/`
- `dry-run/canonical_arxiv_*`
- `dry-run/raw_svg_*`
- `dry-run/strict_ir_seed/`

These are useful for experiments and debugging, but they are not the 50K
multi-source corpus.

## AWS Commands Already Run

Instance: `i-0fe2703b709d15b01` (`svg-finetuning-c7g-dryrun`)

Important SSM command history:

- `2026-05-23 22:45 EDT`, `start cloud wikimedia shards corrected`
  - Created a 32-shard line-aligned plan over
    `scratch/commonswiki-latest-image.sql`.
  - Dump size: `159,526,682,899` bytes.
  - Ran `tools/corpus/run_wikimedia_shards_local.py` with `--limit 7500` and
    `--parallelism 8`.

- `2026-05-24 02:24 EDT`, `merge-wikimedia-ranked-shards`
  - Ran `tools/corpus/merge_wikimedia_ranked_shards.py`.
  - Merged `240,000` shard-local ranked candidates into the global top `7,500`.
  - Uploaded to
    `s3://svg-finetuning-data-446224796301/corpus/wikimedia_dump_ranked_7500/`.

Earlier SSM commands produced older dry-run prefixes. Several starts failed due
to malformed shell heredoc, missing `boto3`, Python 3.9 incompatibility with
`dataclass(slots=True)`, and upstream arXiv/Wikimedia `429/503` responses.

## Standard Commands

Installable console entrypoints are defined in `pyproject.toml` and mirror the
pipeline stages below as `svg-corpus-*` and `svg-train-*` commands.

Write a 50K plan:

```bash
svg-corpus-plan \
  --target-records 50000 \
  --plan-id corpus_plan_50k_v1 \
  --output pipeline_output/corpus/corpus_plan_50k_v1.json
```

Generate synthetic strict IR:

```bash
svg-corpus-build-synthetic \
  --count 10000 \
  --seed 101 \
  --output-root pipeline_output/corpus \
  --corpus-id synthetic_strict_ir_10000_seed101
```

Build a Wikimedia shard plan:

```bash
svg-corpus-build-bulk-candidates \
  --mode wikimedia-shard-plan \
  --wikimedia-dump-path scratch/commonswiki-latest-image.sql \
  --shard-count 32 \
  --shard-plan pipeline_output/corpus/wikimedia_shards_32.json
```

Run Wikimedia shard ranking locally on one machine:

```bash
svg-corpus-run-wikimedia-shards \
  --shard-plan pipeline_output/corpus/wikimedia_shards_32.json \
  --output-dir pipeline_output/corpus/wikimedia_ranked_shards \
  --limit 7500 \
  --parallelism 8
```

Merge Wikimedia shards:

```bash
svg-corpus-merge-wikimedia-shards \
  --ranked-dir pipeline_output/corpus/wikimedia_ranked_shards \
  --corpus-id wikimedia_dump_ranked_7500 \
  --output-root pipeline_output/corpus \
  --dump-path scratch/commonswiki-latest-image.sql \
  --limit 7500
```

Fetch and validate ranked Wikimedia SVG originals:

```bash
AWS_PROFILE=svg-finetuning svg-corpus-fetch-wikimedia \
  --input-candidates s3://svg-finetuning-data-446224796301/corpus/wikimedia_dump_ranked_7500/candidates.jsonl \
  --corpus-id wikimedia_dump_ranked_7500_fetched \
  --output-root pipeline_output/corpus \
  --upload-s3-prefix s3://svg-finetuning-data-446224796301/corpus/wikimedia_dump_ranked_7500_fetched \
  --request-delay-s 1
```

Create an arXiv candidate manifest from a seed list:

```bash
svg-corpus-build-bulk-candidates \
  --source arxiv \
  --corpus-id arxiv_source_seed_v1 \
  --arxiv-id-file pipeline_output/corpus/arxiv_ids.txt
```

This arXiv command only creates a candidate manifest. Use
`tools/corpus/extract_arxiv_sources.py` after source tarballs are available locally
or downloaded by orchestration.

Download requester-pays arXiv source archives from S3:

```bash
AWS_PROFILE=svg-finetuning svg-corpus-download-arxiv \
  --input-candidates pipeline_output/corpus/arxiv_source_seed_v1/candidates.jsonl \
  --destination-root pipeline_output/corpus/arxiv_sources
```

Build GitHub/Common Crawl manifests from exported index rows:

```bash
svg-corpus-build-bulk-candidates \
  --source github \
  --corpus-id github_index_v1 \
  --index-jsonl pipeline_output/corpus/github_svg_index.jsonl \
  --limit 7500

svg-corpus-fetch-commoncrawl-index \
  --crawl-id CC-MAIN-2026-17 \
  --output pipeline_output/corpus/commoncrawl_cc_main_2026_17_svg_index.jsonl \
  --limit 5000

svg-corpus-build-bulk-candidates \
  --source commoncrawl \
  --corpus-id commoncrawl_index_v1 \
  --index-jsonl pipeline_output/corpus/commoncrawl_svg_index.jsonl \
  --limit 5000
```

Extract/fetch, dedupe, normalize, assemble, and evaluate:

```bash
svg-corpus-extract-arxiv \
  --input-candidates pipeline_output/corpus/arxiv_source_seed_v1/candidates.jsonl \
  --source-root pipeline_output/corpus/arxiv_sources \
  --corpus-id arxiv_extracted_v1

svg-corpus-fetch-indexed \
  --input-candidates pipeline_output/corpus/github_index_v1/candidates.jsonl \
  --corpus-id github_fetched_v1

svg-corpus-fetch-indexed \
  --input-candidates pipeline_output/corpus/commoncrawl_index_v1/candidates.jsonl \
  --corpus-id commoncrawl_fetched_v1

svg-corpus-dedupe \
  --corpus-id corpus_deduped_v1 \
  --input-candidates pipeline_output/corpus/synthetic_strict_ir_10000_seed101/candidates.jsonl \
  --input-candidates pipeline_output/corpus/wikimedia_dump_ranked_7500_fetched/candidates.jsonl \
  --input-candidates pipeline_output/corpus/arxiv_extracted_v1/candidates.jsonl

svg-corpus-normalize \
  --input-candidates pipeline_output/corpus/corpus_deduped_v1/candidates.jsonl \
  --corpus-id corpus_normalized_v1

svg-corpus-assemble \
  --dataset-id corpus_50k_v1 \
  --records pipeline_output/corpus/corpus_normalized_v1/training/records.jsonl

svg-corpus-eval \
  --records pipeline_output/datasets/corpus_50k_v1/strict_ir_train/train.jsonl \
  --output pipeline_output/datasets/corpus_50k_v1/eval_report.json

svg-corpus-render \
  --records pipeline_output/datasets/corpus_50k_v1/raw_svg_train/train.jsonl \
  --output-dir pipeline_output/datasets/corpus_50k_v1/eval_visual/render_artifacts

svg-corpus-caption-queue \
  --input-candidates pipeline_output/corpus/corpus_deduped_v1/candidates.jsonl \
  --output pipeline_output/corpus/corpus_deduped_v1/model_caption_queue.jsonl
```

Manual promotion after eval passes:

```bash
AWS_PROFILE=svg-finetuning svg-corpus-promote \
  --product-manifest pipeline_output/datasets/corpus_50k_v1/strict_ir_train/manifest.json \
  --eval-report pipeline_output/datasets/corpus_50k_v1/eval_report.json \
  --data-bucket svg-finetuning-data-446224796301 \
  --s3-dataset-prefix s3://svg-finetuning-data-446224796301/train/corpus_50k_v1/strict_ir_train \
  --approved
```

## What Is Left

1. Finish Wikimedia selective fetch at scale:
   - run `tools/corpus/fetch_validate_wikimedia.py` in batches or on EC2
   - write fetched assets under a new corpus prefix
   - inspect reject/failure reasons
   - keep request delay conservative to avoid Wikimedia rate limits
2. Export real arXiv/GitHub/Common Crawl indexes and run the new workers.
3. Run arXiv S3 downloads, Common Crawl WARC reads, dedupe, normalization,
   assembly, render artifacts, caption queueing, and eval on the full 50K
   target.
4. Review eval and caption-queue outputs before manually promoting a passed
   product manifest.

## Training Safety

Files under `pipeline_output/corpus/**/training/manifest.json` are corpus-local
manifests. They do not overwrite `train/dataset_manifest.json` and do not start
training by themselves. Promotion to a training job is manual until explicitly
approved.
