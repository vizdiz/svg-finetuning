from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json

import boto3


REGION = "us-east-1"
ACCOUNT_ID = "446224796301"
DATA_BUCKET = f"svg-finetuning-data-{ACCOUNT_ID}"
MODELS_BUCKET = f"svg-finetuning-models-{ACCOUNT_ID}"
SCRIPTS_URI = f"s3://svg-finetuning-scripts-{ACCOUNT_ID}/training/"
SM_ROLE = f"arn:aws:iam::{ACCOUNT_ID}:role/SVGFinetuneSageMakerRole"
TRAINING_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="svg-finetuning")
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--job-prefix", default="svg-finetune")
    parser.add_argument("--dataset-manifest-uri", required=True)
    parser.add_argument("--instance-type", default="ml.g5.2xlarge")
    parser.add_argument("--max-length", default="4096")
    parser.add_argument("--max-target-chars", default="100000")
    parser.add_argument("--drop-overlength-records", default="true")
    parser.add_argument("--epochs", default="1")
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--grad-accum", default="8")
    parser.add_argument("--learning-rate", default="2e-4")
    parser.add_argument("--volume-gb", type=int, default=100)
    parser.add_argument("--max-runtime-seconds", type=int, default=86400)
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    secrets = session.client("secretsmanager")
    sm = session.client("sagemaker")
    hf_token = secrets.get_secret_value(SecretId="svg-finetuning/huggingface-token")["SecretString"]
    job_name = f"{args.job_prefix}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

    response = sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=SM_ROLE,
        AlgorithmSpecification={
            "TrainingImage": TRAINING_IMAGE,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["python3", "/opt/ml/input/data/scripts/train.py"],
        },
        HyperParameters={
            "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
            "data_bucket": DATA_BUCKET,
            "dataset_manifest_uri": args.dataset_manifest_uri,
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "learning_rate": args.learning_rate,
            "fp16": "true",
            "max_length": args.max_length,
            "max_prompt_chars": "4096",
            "max_target_chars": args.max_target_chars,
            "drop_overlength_records": args.drop_overlength_records,
            "gradient_checkpointing": "true",
            "lora_r": "16",
            "lora_alpha": "32",
            "lora_dropout": "0.05",
            "lora_target_modules": "q_proj,k_proj,v_proj,o_proj",
            "update_endpoint": "false",
            "endpoint_name": "svg-finetuning-inference",
            "models_bucket": MODELS_BUCKET,
        },
        InputDataConfig=[
            {
                "ChannelName": "scripts",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": SCRIPTS_URI,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{MODELS_BUCKET}/training-jobs/"},
        ResourceConfig={
            "InstanceType": args.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": args.volume_gb,
        },
        StoppingCondition={"MaxRuntimeInSeconds": args.max_runtime_seconds},
        Environment={
            "HF_TOKEN": hf_token,
            "HUGGING_FACE_HUB_TOKEN": hf_token,
            "TRANSFORMERS_CACHE": "/opt/ml/model/.cache",
            "AWS_REGION": args.region,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )

    print(json.dumps({"job_name": job_name, "arn": response["TrainingJobArn"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
