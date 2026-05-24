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

REGION = os.environ.get("AWS_REGION", os.environ.get("REGION", "us-east-1"))
ACCOUNT_ID = os.environ.get("ACCOUNT_ID", "")
DATA_BUCKET = os.environ.get("DATA_BUCKET", f"svg-finetuning-data-{ACCOUNT_ID}" if ACCOUNT_ID else "")
MODELS_BUCKET = os.environ.get("MODELS_BUCKET", f"svg-finetuning-models-{ACCOUNT_ID}" if ACCOUNT_ID else "")
SM_ROLE = os.environ.get("SM_ROLE", f"arn:aws:iam::{ACCOUNT_ID}:role/SVGFinetuneSageMakerRole" if ACCOUNT_ID else "")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "svg-finetuning-inference")
SCRIPTS_URI = os.environ.get("SCRIPTS_URI", f"s3://svg-finetuning-scripts-{ACCOUNT_ID}/training/" if ACCOUNT_ID else "")
TRAINING_INSTANCE_TYPE = os.environ.get("TRAINING_INSTANCE_TYPE", "ml.g5.2xlarge")
TRAINING_IMAGE = os.environ.get(
    "TRAINING_IMAGE",
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04",
)

def get_hf_token() -> str:
    secrets = boto3.client("secretsmanager", region_name=REGION)
    secret_id = os.environ.get("HF_SECRET_ID", "svg-finetuning/huggingface-token")
    return secrets.get_secret_value(SecretId=secret_id)["SecretString"]


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
    if not DATA_BUCKET or not MODELS_BUCKET or not SM_ROLE or not SCRIPTS_URI:
        raise RuntimeError("Lambda env is missing DATA_BUCKET, MODELS_BUCKET, SM_ROLE, or SCRIPTS_URI")

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
            "num_train_epochs":             os.environ.get("TRAINING_EPOCHS", "1"),
            "per_device_train_batch_size":  os.environ.get("TRAINING_BATCH_SIZE", "1"),
            "gradient_accumulation_steps":  os.environ.get("TRAINING_GRAD_ACCUM", "8"),
            "learning_rate":                "2e-4",
            "fp16":                         "true",
            "max_length":                   os.environ.get("TRAINING_MAX_LENGTH", "4096"),
            "max_prompt_chars":             os.environ.get("TRAINING_MAX_PROMPT_CHARS", "4096"),
            "max_target_chars":             os.environ.get("TRAINING_MAX_TARGET_CHARS", "100000"),
            "drop_overlength_records":       os.environ.get("TRAINING_DROP_OVERLENGTH_RECORDS", "true"),
            "gradient_checkpointing":       os.environ.get("TRAINING_GRADIENT_CHECKPOINTING", "true"),
            "lora_r":                       "16",
            "lora_alpha":                   "32",
            "lora_dropout":                 "0.05",
            "lora_target_modules":          "q_proj,k_proj,v_proj,o_proj",
            "update_endpoint":              os.environ.get("TRAINING_UPDATE_ENDPOINT", "false"),
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
            "InstanceType":    TRAINING_INSTANCE_TYPE,
            "InstanceCount":   1,
            "VolumeSizeInGB":  100,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 86400},
        Environment={
            "HF_TOKEN":               hf_token,
            "HUGGING_FACE_HUB_TOKEN": hf_token,
            "TRANSFORMERS_CACHE":     "/opt/ml/model/.cache",
            "AWS_REGION":             REGION,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )

    print(f"Training job started: {job_name}")
    return {
        "statusCode": 200,
        "body": json.dumps({"jobName": job_name, "arn": resp["TrainingJobArn"]}),
    }
