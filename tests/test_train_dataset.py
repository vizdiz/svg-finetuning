from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

datasets_module = ModuleType("datasets")


class _Dataset(list):
    @classmethod
    def from_dict(cls, payload):
        rows = []
        keys = list(payload)
        for index in range(len(payload[keys[0]])):
            rows.append({key: payload[key][index] for key in keys})
        return cls(rows)


datasets_module.Dataset = _Dataset
transformers_module = ModuleType("transformers")
transformers_module.AutoTokenizer = SimpleNamespace()
transformers_module.AutoModelForCausalLM = SimpleNamespace()
transformers_module.TrainingArguments = object
transformers_module.Trainer = object
transformers_module.DataCollatorForSeq2Seq = object
peft_module = ModuleType("peft")
peft_module.LoraConfig = object
peft_module.get_peft_model = lambda model, config: model
peft_module.TaskType = SimpleNamespace(CAUSAL_LM="CAUSAL_LM")

sys.modules.setdefault("torch", SimpleNamespace(float16="float16", float32="float32"))
sys.modules.setdefault("datasets", datasets_module)
sys.modules.setdefault("transformers", transformers_module)
sys.modules.setdefault("peft", peft_module)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "training"))

from backend.training import train
from backend.training.dataset_interface import DatasetManifest, TrainingRecord


class _Tokenizer:
    pad_token_id = 0

    def __init__(self):
        self.seen_user_texts: list[str] = []
        self.seen_assistant_texts: list[str] = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        for message in messages:
            if message["role"] == "user":
                self.seen_user_texts.append(message["content"])
            if message["role"] == "assistant":
                self.seen_assistant_texts.append(message["content"])
        text = "\n".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return list(range(max(1, len(text))))


class _Loader:
    def __init__(self, manifest):
        self.manifest = manifest

    def iter_records(self):
        yield TrainingRecord(
            id="huge",
            prompt="p" * 100,
            diagram_ir={"nodes": [{"id": "a", "kind": "box", "label": "x" * 5000}], "edges": []},
        )


def test_build_dataset_truncates_before_tokenization(monkeypatch):
    monkeypatch.setattr(train, "DatasetLoader", _Loader)
    tokenizer = _Tokenizer()
    manifest = DatasetManifest(
        dataset_id="unit",
        created_at="2026-05-22T00:00:00Z",
        record_count=1,
        files=["s3://bucket/train.jsonl"],
    )

    dataset = train.build_dataset(
        tokenizer,
        manifest,
        max_length=32,
        max_prompt_chars=12,
        max_target_chars=64,
        drop_overlength_records=False,
    )

    assert len(dataset) == 1
    assert max(len(text) for text in tokenizer.seen_user_texts) == 12
    assert max(len(text) for text in tokenizer.seen_assistant_texts) == 64
    assert len(dataset[0]["input_ids"]) == 32


def test_build_dataset_skips_overlength_records_by_default(monkeypatch):
    monkeypatch.setattr(train, "DatasetLoader", _Loader)
    tokenizer = _Tokenizer()
    manifest = DatasetManifest(
        dataset_id="unit",
        created_at="2026-05-22T00:00:00Z",
        record_count=1,
        files=["s3://bucket/train.jsonl"],
    )

    try:
        train.build_dataset(
            tokenizer,
            manifest,
            max_length=32,
            max_prompt_chars=12,
            max_target_chars=64,
        )
    except ValueError as exc:
        assert "No records fit max_length=32" in str(exc)
    else:
        raise AssertionError("expected overlength dataset to be rejected")
