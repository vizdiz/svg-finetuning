"""
train.py — LoRA fine-tuning for SVG generation on Qwen2.5-7B-Instruct.
Runs inside the SageMaker HuggingFace PyTorch training container.

Dataset coupling: reads a DatasetManifest from S3, delegates all data
loading to DatasetLoader. The dataset pipeline is a separate concern —
see dataset_interface.py for the contract.

Qwen2.5-specific details:
  - Requires trust_remote_code=True for tokenizer and model.
  - Uses the Qwen2.5 chat template (<|im_start|>/<|im_end|>) so the model
    learns the correct input/output structure.
  - Labels are masked over the system + user turns so loss is computed
    only on the assistant response.
  - LoRA targets q_proj, k_proj, v_proj, o_proj (all attention projections).
"""
import json
import os
import time
import boto3
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

from dataset_interface import DatasetLoader, DatasetManifest, read_manifest_from_s3

SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
SM_HP_PATH   = "/opt/ml/input/config/hyperparameters.json"
REGION       = os.environ.get("AWS_REGION",   "us-east-1")

SYSTEM_PROMPT_SVG = "You are an SVG generation expert. Given a description, produce well-formed SVG markup."
SYSTEM_PROMPT_IR = "You are a diagram compiler. Given a description, produce valid diagram IR JSON only."


def load_hyperparameters() -> dict:
    with open(SM_HP_PATH) as f:
        raw = json.load(f)
    return {
        "model_name_or_path":          raw.get("model_name_or_path", "Qwen/Qwen2.5-7B-Instruct"),
        "data_bucket":                  raw["data_bucket"],
        "dataset_manifest_uri":         raw.get("dataset_manifest_uri", ""),
        "num_train_epochs":        int(raw.get("num_train_epochs", "3")),
        "per_device_train_batch_size": int(raw.get("per_device_train_batch_size", "2")),
        "gradient_accumulation_steps": int(raw.get("gradient_accumulation_steps", "4")),
        "learning_rate":           float(raw.get("learning_rate", "2e-4")),
        "fp16":                          raw.get("fp16", "true").lower() == "true",
        "max_length":               int(raw.get("max_length", "2048")),
        "max_prompt_chars":         int(raw.get("max_prompt_chars", "4096")),
        "max_target_chars":         int(raw.get("max_target_chars", "100000")),
        "gradient_checkpointing":        raw.get("gradient_checkpointing", "true").lower() == "true",
        "drop_overlength_records":       raw.get("drop_overlength_records", "true").lower() == "true",
        "lora_r":                   int(raw.get("lora_r", "16")),
        "lora_alpha":               int(raw.get("lora_alpha", "32")),
        "lora_dropout":            float(raw.get("lora_dropout", "0.05")),
        "lora_target_modules":           raw.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj").split(","),
        "endpoint_name":                 raw.get("endpoint_name", ""),
        "models_bucket":                 raw.get("models_bucket", ""),
        "update_endpoint":               raw.get("update_endpoint", "false").lower() == "true",
}


def read_manifest_from_uri(uri: str) -> DatasetManifest:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// manifest URI, got: {uri}")
    bucket, key = uri[len("s3://"):].split("/", 1)
    s3 = boto3.client("s3", region_name=REGION)
    raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    manifest = DatasetManifest.from_json(raw)
    manifest.validate()
    return manifest


def _truncate_text(text: str, max_chars: int, label: str, record_id: str) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    print(f"Truncating {label} for record {record_id or '<unknown>'}: {len(text)} -> {max_chars} chars")
    return text[:max_chars]


def _target_text(record) -> str:
    if record.diagram_ir is not None:
        # Compact JSON matters for SVG-extracted IR; pretty-printing can add
        # enough tokens to blow up a dry-run job before max_length truncation.
        return json.dumps(record.diagram_ir, sort_keys=True, separators=(",", ":"))
    return record.target_text()


def build_dataset(
    tokenizer,
    manifest,
    max_length: int,
    max_prompt_chars: int = 4096,
    max_target_chars: int = 12000,
    drop_overlength_records: bool = True,
) -> Dataset:
    """
    Formats each record using the Qwen2.5 chat template and masks prompt
    tokens in labels so loss is computed only on the assistant output.

    IR-labeled records train the model to emit JSON. Legacy SVG records
    remain supported for backward compatibility.
    """
    loader = DatasetLoader(manifest)
    input_ids_list, attention_mask_list, labels_list = [], [], []
    skipped_overlength = 0

    for record in loader.iter_records():
        system_prompt = SYSTEM_PROMPT_IR if record.diagram_ir is not None else SYSTEM_PROMPT_SVG
        prompt_text = _truncate_text(record.prompt, max_prompt_chars, "prompt", record.id)
        target = _target_text(record)
        target_text = _truncate_text(target, max_target_chars, "target", record.id)

        # Tokenize the prompt-only portion to determine the mask boundary.
        # add_generation_prompt=True appends <|im_start|>assistant\n so the
        # boundary lands exactly where the model should start generating.
        prompt_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )

        # Full sequence: system + user + assistant (SVG or IR JSON) + end token
        full_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": target_text},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )

        if len(full_ids) > max_length:
            if drop_overlength_records:
                skipped_overlength += 1
                print(
                    "Skipping overlength record "
                    f"{record.id or '<unknown>'}: prompt_tokens={len(prompt_ids)} "
                    f"full_tokens={len(full_ids)} max_length={max_length}"
                )
                continue
            print(
                "Token-truncating overlength record "
                f"{record.id or '<unknown>'}: full_tokens={len(full_ids)} max_length={max_length}"
            )
            full_ids = full_ids[:max_length]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        attention_mask = [1] * len(full_ids)

        input_ids_list.append(full_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    if not input_ids_list:
        raise ValueError(f"No records fit max_length={max_length}; refusing to train on truncated targets")

    if skipped_overlength:
        print(
            f"Skipped {skipped_overlength} overlength record(s); "
            f"kept {len(input_ids_list)} full-target record(s)"
        )

    return Dataset.from_dict({
        "input_ids":      input_ids_list,
        "attention_mask": attention_mask_list,
        "labels":         labels_list,
        "length":         [len(input_ids) for input_ids in input_ids_list],
    })


def update_endpoint(endpoint_name: str, model_s3_uri: str, models_bucket: str, account_id: str) -> None:
    sm  = boto3.client("sagemaker", region_name=REGION)
    ts  = str(int(time.time()))
    lmi = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/djl-inference:0.36.0-lmi25.0.0-cu130"

    model_name  = f"svg-finetuning-vllm-model-{ts}"
    config_name = f"svg-finetuning-async-endpoint-config-{ts}"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image":        lmi,
            "ModelDataUrl": model_s3_uri,
            "Environment": {
                "OPTION_ROLLING_BATCH":          "vllm",
                "OPTION_TENSOR_PARALLEL_DEGREE": "1",
                "OPTION_DTYPE":                  "fp16",
                "OPTION_MAX_ROLLING_BATCH_SIZE": "32",
                "OPTION_TRUST_REMOTE_CODE":      "true",
            },
        },
        ExecutionRoleArn=f"arn:aws:iam::{account_id}:role/SVGFinetuneSageMakerRole",
    )
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName":          "AllTraffic",
            "ModelName":            model_name,
            "InstanceType":         "ml.g5.2xlarge",
            "InitialInstanceCount": 1,
        }],
        AsyncInferenceConfig={
            "OutputConfig": {
                "S3OutputPath": f"s3://{models_bucket}/async-inference/output/"
            },
            "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 4},
        },
    )
    sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    print(f"Endpoint '{endpoint_name}' update triggered → model '{model_name}'")


def main():
    hp = load_hyperparameters()
    print(f"Hyperparameters: {json.dumps({k: v for k, v in hp.items() if k != 'data_bucket'}, indent=2)}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        hp["model_name_or_path"],
        trust_remote_code=True,
        padding_side="right",
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    if hp["dataset_manifest_uri"]:
        print(f"Reading manifest from {hp['dataset_manifest_uri']}")
        manifest = read_manifest_from_uri(hp["dataset_manifest_uri"])
    else:
        print(f"Reading manifest from s3://{hp['data_bucket']}/train/dataset_manifest.json")
        manifest = read_manifest_from_s3(hp["data_bucket"])
    print(f"Dataset: {manifest.dataset_id}, {manifest.record_count} records, {len(manifest.files)} file(s)")

    dataset = build_dataset(
        tokenizer,
        manifest,
        hp["max_length"],
        max_prompt_chars=hp["max_prompt_chars"],
        max_target_chars=hp["max_target_chars"],
        drop_overlength_records=hp["drop_overlength_records"],
    )
    print(f"Tokenized dataset: {len(dataset)} examples")

    # ── Base model ───────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        hp["model_name_or_path"],
        trust_remote_code=True,
        torch_dtype=torch.float16 if hp["fp16"] else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False
    if hp["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=hp["lora_dropout"],
        target_modules=hp["lora_target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=SM_MODEL_DIR,
            num_train_epochs=hp["num_train_epochs"],
            per_device_train_batch_size=hp["per_device_train_batch_size"],
            gradient_accumulation_steps=hp["gradient_accumulation_steps"],
            learning_rate=hp["learning_rate"],
            fp16=hp["fp16"],
            save_strategy="epoch",
            logging_steps=10,
            report_to="none",
            dataloader_num_workers=4,
            remove_unused_columns=False,
            group_by_length=True,
            length_column_name="length",
            gradient_checkpointing=hp["gradient_checkpointing"],
        ),
        train_dataset=dataset,
        # DataCollatorForSeq2Seq handles the pre-computed labels correctly
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=False),
    )
    trainer.train()

    # ── Merge LoRA into base model and save ───────────────────────────────────
    print("Merging LoRA weights...")
    model.merge_and_unload().save_pretrained(SM_MODEL_DIR)
    tokenizer.save_pretrained(SM_MODEL_DIR)

    # ── Update inference endpoint ─────────────────────────────────────────────
    if hp["update_endpoint"] and hp["endpoint_name"] and hp["models_bucket"]:
        job_name   = os.environ.get("TRAINING_JOB_NAME", "unknown")
        account_id = boto3.client("sts").get_caller_identity()["Account"]
        model_s3   = f"s3://{hp['models_bucket']}/training-jobs/{job_name}/output/model.tar.gz"
        update_endpoint(hp["endpoint_name"], model_s3, hp["models_bucket"], account_id)


if __name__ == "__main__":
    main()
