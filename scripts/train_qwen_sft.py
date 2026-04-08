from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class CompletionOnlyDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int, limit: int | None = None):
        self.samples: list[dict[str, list[int]]] = []
        self.skipped_too_long = 0
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if limit is not None and len(self.samples) >= limit:
                    break
                row = json.loads(line)
                prompt_ids = tokenizer(row["prompt_text"], add_special_tokens=False)["input_ids"]
                completion_ids = tokenizer(row["completion_text"], add_special_tokens=False)["input_ids"]
                completion_ids.append(tokenizer.eos_token_id)
                input_ids = prompt_ids + completion_ids
                labels = ([-100] * len(prompt_ids)) + completion_ids
                if len(input_ids) > max_length:
                    self.skipped_too_long += 1
                    continue
                self.samples.append(
                    {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": [1] * len(input_ids),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.samples[index]


@dataclass
class CompletionCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + ([self.pad_token_id] * pad))
            batch["attention_mask"].append(feature["attention_mask"] + ([0] * pad))
            batch["labels"].append(feature["labels"] + ([-100] * pad))
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def build_quantization_config(cfg: dict) -> BitsAndBytesConfig | None:
    if not cfg.get("load_in_4bit", True):
        return None
    compute_dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=cfg.get("trust_remote_code", False))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = CompletionOnlyDataset(
        Path(cfg["train_file"]),
        tokenizer,
        cfg["max_length"],
        limit=cfg.get("max_train_samples"),
    )
    eval_dataset = CompletionOnlyDataset(
        Path(cfg["val_file"]),
        tokenizer,
        cfg["max_length"],
        limit=cfg.get("max_eval_samples"),
    )

    quantization_config = build_quantization_config(cfg)
    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        dtype=dtype,
        trust_remote_code=cfg.get("trust_remote_code", False),
        quantization_config=quantization_config,
        device_map=cfg.get("device_map", "auto"),
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        warmup_steps=cfg.get("warmup_steps", 0),
        weight_decay=cfg.get("weight_decay", 0.0),
        logging_steps=cfg.get("logging_steps", 10),
        eval_steps=cfg.get("eval_steps", 100),
        save_steps=cfg.get("save_steps", 100),
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=cfg.get("save_total_limit", 2),
        bf16=cfg.get("bf16", True),
        fp16=cfg.get("fp16", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        seed=cfg.get("seed", 1337),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_state()

    best_dir = out_dir / "best_adapter"
    final_dir = out_dir / "final_adapter"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(best_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.model.save_pretrained(final_dir)

    summary = {
        "base_model": cfg["base_model"],
        "train_samples": len(train_dataset),
        "val_samples": len(eval_dataset),
        "train_skipped_too_long": train_dataset.skipped_too_long,
        "val_skipped_too_long": eval_dataset.skipped_too_long,
        "max_length": cfg["max_length"],
        "train_metrics": train_result.metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "config": cfg,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
