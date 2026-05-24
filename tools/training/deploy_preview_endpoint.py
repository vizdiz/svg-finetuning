from __future__ import annotations

import argparse
import time

import boto3
from botocore.exceptions import ClientError


REGION = "us-east-1"
DEFAULT_MODEL_DATA = (
    "s3://svg-finetuning-models-446224796301/"
    "training-jobs/svg-finetune-20260522-201153/output/model.tar.gz"
)
DEFAULT_ENDPOINT = "svg-finetuning-inference"
DEFAULT_ROLE = "arn:aws:iam::446224796301:role/SVGFinetuneSageMakerRole"
LMI_IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.36.0-lmi25.0.0-cu130"


def endpoint_exists(sm, endpoint_name: str) -> bool:
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ValidationException":
            return False
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the dry-run SVG model as a SageMaker preview endpoint.")
    parser.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model-data", default=DEFAULT_MODEL_DATA)
    parser.add_argument("--role-arn", default=DEFAULT_ROLE)
    parser.add_argument("--instance-type", default="ml.g5.2xlarge")
    parser.add_argument("--region", default=REGION)
    args = parser.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    ts = int(time.time())
    model_name = f"svg-preview-model-{ts}"
    config_name = f"svg-preview-endpoint-config-{ts}"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": LMI_IMAGE,
            "ModelDataUrl": args.model_data,
            "Environment": {
                "OPTION_MODEL_ID": "/opt/ml/model",
                "OPTION_ROLLING_BATCH": "vllm",
                "OPTION_TENSOR_PARALLEL_DEGREE": "1",
                "OPTION_DTYPE": "fp16",
                "OPTION_MAX_MODEL_LEN": "4096",
                "OPTION_MAX_ROLLING_BATCH_SIZE": "4",
                "OPTION_TRUST_REMOTE_CODE": "true",
            },
        },
        ExecutionRoleArn=args.role_arn,
    )

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": args.instance_type,
                "InitialInstanceCount": 1,
            }
        ],
    )

    if endpoint_exists(sm, args.endpoint_name):
        sm.update_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=config_name)
        action = "update"
    else:
        sm.create_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=config_name)
        action = "create"

    print(
        f"Started endpoint {action}: endpoint={args.endpoint_name} "
        f"config={config_name} model={model_name} instance={args.instance_type}"
    )


if __name__ == "__main__":
    main()
