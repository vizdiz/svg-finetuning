# svg-finetuning

SVG generation model — fine-tuning infrastructure on AWS.

## Architecture

- **Training**: SageMaker training jobs (LoRA fine-tuning via PEFT on Qwen2.5-7B)
- **Inference**: SageMaker async endpoint with vLLM (DJL LMI container), hosted entirely on AWS
- **Trigger**: Lambda fires on `train/dataset_manifest.json` upload or weekly EventBridge cron

## Structure

```
dataset_pipeline/           # Scrapes SVGs from arXiv & Wikipedia, captions with Claude
  scrapers/                 # Source-specific scrapers (arXiv, Wikipedia)
  processing/               # Validate, normalise, and caption raw SVGs
  pipeline/                 # Orchestration and manifest writing
  config.py                 # All settings via environment variables
  requirements.txt

training/
  dataset_interface.py      # Contract between dataset pipeline and training (DatasetManifest, DatasetLoader)
  train.py                  # LoRA fine-tuning script (runs inside SageMaker container)

lambda/
  lambda_function.py        # Retraining trigger — validates manifest, launches SageMaker job
```

## Dataset contract

The dataset pipeline writes:
1. JSONL files with `{"prompt": "...", "svg": "..."}` records to S3
2. A `DatasetManifest` to `s3://svg-finetuning-data-<account>/train/dataset_manifest.json`

Writing the manifest triggers training. See `training/dataset_interface.py` for the full schema.

## Setup

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY
pip install -r dataset_pipeline/requirements.txt
```

## Deploying changes

```bash
# Update training scripts
AWS_PROFILE=svg-finetuning aws s3 sync training/ \
  s3://svg-finetuning-scripts-446224796301/training/ --region us-east-1

# Update Lambda
cd lambda && zip function.zip lambda_function.py
AWS_PROFILE=svg-finetuning aws lambda update-function-code \
  --function-name svg-finetuning-retraining-trigger \
  --zip-file fileb://function.zip --region us-east-1
cd ..
```

## AWS resources

- **Account**: 446224796301 | **Region**: us-east-1
- **IAM user**: `claude-code-svg-finetuning` (`AWS_PROFILE=svg-finetuning`)
- **S3**: `svg-finetuning-{data,models,logs,scripts}-446224796301`
- **Lambda**: `svg-finetuning-retraining-trigger`
- **Endpoint config**: `svg-finetuning-async-endpoint-config` (ml.g5.2xlarge, async)
- **VPC**: `vpc-077923eb5f22d2bac` (10.0.0.0/16)
