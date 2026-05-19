"""
svg-finetuning-retraining-trigger
Triggered by:
  - S3: dataset pipeline writes train/dataset_manifest.json to the data bucket
  - EventBridge: weekly fallback (cron Sun 2am UTC)

Reads the manifest from S3 to confirm it's valid before launching training.
All dataset concerns live in dataset_interface.py — this function knows
nothing about data format or generation.
"""
import json
import os
import boto3
from datetime import datetime, timezone

ACCOUNT_ID    = os.environ["ACCOUNT_ID"]
REGION        = os.environ.get("AWS_REGION", "us-east-1")
DATA_BUCKET   = f"svg-finetuning-data-{ACCOUNT_ID}"
MODELS_BUCKET = f"svg-finetuning-models-{ACCOUNT_ID}"
SM_ROLE       = f"arn:aws:iam::{ACCOUNT_ID}:role/SVGFinetuneSageMakerRole"
ENDPOINT_NAME = "svg-finetuning-inference"
SCRIPTS_URI   = f"s3://svg-finetuning-scripts-{ACCOUNT_ID}/training/"

TRAINING_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04"
)


def get_hf_token() -> str:
    secrets = boto3.client("secretsmanager", region_name=REGION)
    return secrets.get_secret_value(
        SecretId="svg-finetuning/huggingface-token"
    )["SecretString"]


def validate_manifest(bucket: str) -> dict:
    """
    Read and validate the manifest before launching training.
    Raises if the manifest is missing or malformed — prevents training
    jobs from starting against incomplete datasets.
    """
    s3  = boto3.client("s3", region_name=REGION)
    raw = s3.get_object(Bucket=bucket, Key="train/dataset_manifest.json")["Body"].read()
    m   = json.loads(raw)

    required = {"dataset_id", "record_count", "files"}
    missing  = required - m.keys()
    if missing:
        raise ValueError(f"Manifest missing fields: {missing}")
    if not m["files"]:
        raise ValueError("Manifest lists no data files")
    if m["record_count"] <= 0:
        raise ValueError(f"Manifest record_count={m['record_count']}")

    return m


def handler(event, context):
    sm = boto3.client("sagemaker", region_name=REGION)

    # Validate manifest regardless of trigger source
    manifest = validate_manifest(DATA_BUCKET)
    print(f"Manifest OK: dataset_id={manifest['dataset_id']}, records={manifest['record_count']}")

    hf_token = get_hf_token()
    job_name = f"svg-finetune-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    resp = sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=SM_ROLE,
        AlgorithmSpecification={
            "TrainingImage":      TRAINING_IMAGE,
            "TrainingInputMode":  "File",
            # train.py and dataset_interface.py are injected via the scripts channel
            "ContainerEntrypoint": ["python3", "/opt/ml/input/data/scripts/train.py"],
        },
        HyperParameters={
            "model_name_or_path":          "Qwen/Qwen2.5-7B-Instruct",
            # data_bucket is the only dataset coupling — train.py reads the manifest from it
            "data_bucket":                  DATA_BUCKET,
            "num_train_epochs":             "3",
            "per_device_train_batch_size":  "2",
            "gradient_accumulation_steps":  "4",
            "learning_rate":                "2e-4",
            "fp16":                         "true",
            "lora_r":                       "16",
            "lora_alpha":                   "32",
            "lora_dropout":                 "0.05",
            "lora_target_modules":          "q_proj,k_proj,v_proj,o_proj",
            "endpoint_name":                ENDPOINT_NAME,
            "models_bucket":                MODELS_BUCKET,
        },
        InputDataConfig=[
            {
                "ChannelName": "scripts",
                "DataSource":  {"S3DataSource": {
                    "S3DataType":              "S3Prefix",
                    "S3Uri":                   SCRIPTS_URI,
                    "S3DataDistributionType":  "FullyReplicated",
                }},
            },
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{MODELS_BUCKET}/training-jobs/"},
        ResourceConfig={
            "InstanceType":    "ml.g5.2xlarge",
            "InstanceCount":   1,
            "VolumeSizeInGB":  100,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 86400},
        Environment={
            "HF_TOKEN":               hf_token,
            "HUGGING_FACE_HUB_TOKEN": hf_token,
            "TRANSFORMERS_CACHE":     "/opt/ml/model/.cache",
        },
    )

    print(f"Training job started: {job_name}")
    return {
        "statusCode": 200,
        "body": json.dumps({"jobName": job_name, "arn": resp["TrainingJobArn"]}),
    }
