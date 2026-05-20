# svg-finetuning

SVG generation model — fine-tuning, inference, feedback, and deployment infrastructure on AWS.

## Architecture

- **Dataset pipeline**: scrapes SVGs from arXiv and Wikimedia, validates, ranks, captions, and writes manifests
- **Training**: SageMaker processing/training/evaluation pipeline definitions and model registry helpers
- **Inference**: SageMaker-hosted custom vLLM request router with a cache-first serving path
- **Feedback**: JSONL schema + ingestion endpoint + rate-before-download gate
- **Public API**: Lambda + API Gateway + CloudFront `/api/*` routing with session persistence
- **Trigger**: Lambda fires on `train/dataset_manifest.json` upload or weekly EventBridge cron

## Structure

```
backend/
  dataset_pipeline/         # Scrapes SVGs from arXiv & Wikimedia, validates, captions, writes manifests
    scrapers/               # Source-specific scrapers (arXiv, Wikimedia)
    processing/             # Validate, normalise, and caption raw SVGs
    pipeline/               # Orchestration, manifest writing, pipeline definitions
    config.py               # All settings via environment variables
    requirements.txt

  training/
    dataset_interface.py    # Contract between dataset pipeline and training (DatasetManifest, DatasetLoader)
    preprocess.py           # SageMaker processing stub
    train.py                # LoRA fine-tuning script (runs inside SageMaker container)
    evaluate.py             # SageMaker evaluation stub
    pipeline_definition.py  # DAG validation / pipeline spec helpers
    model_registry.py       # Model Package Group helpers

  inference/
    router.py               # API Gateway -> cache -> endpoint router
    app.py                   # Lambda/API Gateway service layer and session persistence
    sessions.py              # Session record models + local/DynamoDB stores
    contracts.py             # Internal JSON contract for the SageMaker vLLM endpoint
    endpoint_clients.py      # SageMaker runtime client for the custom vLLM endpoint
    vllm/                   # Custom SageMaker vLLM container

  feedback/
    schema.py               # JSONL feedback schema
    gating.py               # rate-before-download gate logic
    ingest.py               # Lambda-friendly feedback ingestion entrypoint

lambda/
  lambda_function.py        # Retraining trigger — validates manifest, launches SageMaker job
```

## Dataset contract

The dataset pipeline writes JSONL records with the prompt, target SVG, and optional `diagram_ir` metadata to S3, plus a `DatasetManifest` to `s3://svg-finetuning-data-<account>/train/dataset_manifest.json`.

Writing the manifest triggers training. See `backend/training/dataset_interface.py` for the full schema.

## Setup

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY
pip install -r backend/dataset_pipeline/requirements.txt
```

## Deploying changes

```bash
# Update training scripts
AWS_PROFILE=<your-profile> aws s3 sync backend/training/ \
  s3://<scripts-bucket>/training/ --region <region>

# Update Lambda
cd lambda && zip function.zip lambda_function.py
AWS_PROFILE=<your-profile> aws lambda update-function-code \
  --function-name <lambda-name> \
  --zip-file fileb://function.zip --region <region>
cd ..
```

## Infra

Terraform lives in `infra/terraform/` and provisions the S3 buckets, Lambda trigger, SageMaker execution role, and weekly fallback schedule.

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

## AWS resources

- AWS-backed storage, training, inference, and retraining infrastructure
- Environment-specific resource names and account details are intentionally omitted from the public README
- The public API surface is routed through CloudFront to API Gateway and a Lambda handler backed by DynamoDB session state
