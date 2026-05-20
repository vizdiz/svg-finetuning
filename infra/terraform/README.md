# Terraform Infra

This directory provisions the AWS resources used by the dataset pipeline:

- S3 buckets for data, models, scripts, and logs
- Lambda retraining trigger
- Public API Lambda, DynamoDB session store, and API Gateway front door
- SageMaker execution role
- S3 manifest trigger
- Weekly EventBridge fallback

## Apply

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

If you use an AWS profile:

```bash
AWS_PROFILE=svg-finetuning terraform plan
AWS_PROFILE=svg-finetuning terraform apply
```

## Expected flow

1. The dataset pipeline writes `train/dataset_manifest.json` to the data bucket.
2. S3 invokes the Lambda trigger.
3. Lambda validates the manifest and starts a SageMaker training job.
4. A weekly EventBridge rule can re-run the same Lambda as a fallback.
5. CloudFront forwards `/api/*` to API Gateway, which invokes the public API Lambda and persists session state in DynamoDB.
6. The public API Lambda invokes the SageMaker custom vLLM endpoint using the repo's internal JSON contract.

## Notes

- Bucket names are derived from the current AWS account id.
- The SageMaker execution role is intentionally narrow and only grants access to the data, scripts, models, and logs buckets.
- The Lambda is packaged directly from `lambda/lambda_function.py` using Terraform's `archive_file` provider.
