import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    train_file: str = "merged_arabic_english_tutor_train_weighted_gold3x.jsonl"
    val_file: str = "merged_arabic_english_tutor_val.jsonl"
    output_dir: str = "qwen25_german_tutor_qlora"
    max_length: int = 2048
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    seed: int = 42
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    report_to: str = "none"
    optim: str = "adamw_hf"
    trust_remote_code: bool = False


def load_config(path: str | None) -> Config:
    cfg = Config()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for k, v in loaded.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--report_to", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for k, v in vars(args).items():
        if k != "config" and v is not None:
            setattr(cfg, k, v)
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_tokenizer(model_name: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def render_segments(messages):
    segments = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role in {"system", "user"}:
            segments.append((f"<|im_start|>{role}\n{content}<|im_end|>\n", False))
        elif role == "assistant":
            segments.append((f"<|im_start|>assistant\n", False))
            segments.append((f"{content}<|im_end|>\n", True))
        else:
            raise ValueError(f"Unsupported role: {role}")
    return segments


def tokenize_example(example, tokenizer, max_length):
    input_ids = []
    labels = []
    for text, trainable in render_segments(example["messages"]):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        labels.extend(ids if trainable else [-100] * len(ids))
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids[:max_length],
        "labels": labels[:max_length],
        "attention_mask": attention_mask[:max_length],
        "seq_len": min(len(input_ids), max_length),
        "was_truncated": int(len(input_ids) > max_length),
    }


class DataCollatorForCausalLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for x in features:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    tokenizer = build_tokenizer(cfg.model_name, cfg.trust_remote_code)
    torch_dtype = torch.bfloat16 if cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
    )
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    raw = load_dataset(
        "json",
        data_files={"train": cfg.train_file, "validation": cfg.val_file},
    )
    tokenized = raw.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.max_length},
        remove_columns=raw["train"].column_names,
        desc="Tokenizing",
    )

    train_truncated = sum(tokenized["train"]["was_truncated"])
    val_truncated = sum(tokenized["validation"]["was_truncated"])
    stats = {
        "train_examples": len(tokenized["train"]),
        "validation_examples": len(tokenized["validation"]),
        "train_truncated": int(train_truncated),
        "validation_truncated": int(val_truncated),
        "max_train_seq_len": int(max(tokenized["train"]["seq_len"])),
        "max_val_seq_len": int(max(tokenized["validation"]["seq_len"])),
    }
    with open(output_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        bf16=cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=cfg.fp16 or (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        report_to=[] if cfg.report_to == "none" else [cfg.report_to],
        optim=cfg.optim,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_pin_memory=True,
        logging_first_step=True,
        group_by_length=True,
        save_safetensors=True,
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForCausalLM(tokenizer),
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    metrics = train_result.metrics
    metrics.update(trainer.evaluate())
    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
