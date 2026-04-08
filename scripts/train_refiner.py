from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from poetry_lm.refiner import MaskedRefiner, RefinerConfig, corrupt_ids
from poetry_lm.tokenizer import Tokenizer
from poetry_lm.training import load_config


def autocast_context(device: str, dtype_name: str):
    if device == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    return torch.autocast(device_type="cuda", dtype=dtype)


@dataclass
class EncodedSample:
    ids: list[int]


class RefinerDataset(Dataset):
    def __init__(self, path: Path, tokenizer: Tokenizer, max_length: int, limit: int | None = None):
        self.samples: list[EncodedSample] = []
        skipped = 0
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                formatted = row.get("formatted_text")
                if not formatted:
                    continue
                ids = tokenizer.encode_formatted(formatted)
                if len(ids) > max_length:
                    skipped += 1
                    continue
                self.samples.append(EncodedSample(ids=ids))
                if limit and len(self.samples) >= limit:
                    break
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> EncodedSample:
        return self.samples[idx]


class RefinerCollator:
    def __init__(
        self,
        tokenizer: Tokenizer,
        mask_prob: float,
        tail_span: int,
        tail_boost: float,
        pad_to_multiple_of: int = 8,
    ):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.tail_span = tail_span
        self.tail_boost = tail_boost
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: list[EncodedSample]) -> dict[str, torch.Tensor]:
        masked_batch: list[list[int]] = []
        target_batch: list[list[int]] = []
        loss_mask_batch: list[list[int]] = []
        max_len = 0
        for sample in batch:
            masked, targets, loss_mask = corrupt_ids(
                sample.ids,
                tokenizer=self.tokenizer,
                mask_prob=self.mask_prob,
                tail_span=self.tail_span,
                tail_boost=self.tail_boost,
            )
            masked_batch.append(masked)
            target_batch.append(targets)
            loss_mask_batch.append(loss_mask)
            max_len = max(max_len, len(masked))
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        pad_id = self.tokenizer.sp.pad_id()
        x, y, m = [], [], []
        for masked, targets, loss_mask in zip(masked_batch, target_batch, loss_mask_batch):
            pad = max_len - len(masked)
            x.append(masked + [pad_id] * pad)
            y.append(targets + [pad_id] * pad)
            m.append(loss_mask + [0] * pad)
        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "targets": torch.tensor(y, dtype=torch.long),
            "loss_mask": torch.tensor(m, dtype=torch.float32),
        }


@torch.no_grad()
def estimate_loss(model: MaskedRefiner, loader: DataLoader, cfg: dict, device: str) -> float:
    model.eval()
    losses = []
    for idx, batch in enumerate(loader):
        if idx >= cfg["eval_steps"]:
            break
        x = batch["input_ids"].to(device)
        y = batch["targets"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        with autocast_context(device, cfg["dtype"]):
            _, loss = model(x, y, loss_mask)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def make_loader(dataset: RefinerDataset, cfg: dict, tokenizer: Tokenizer, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"] if shuffle else cfg.get("eval_batch_size", cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=RefinerCollator(
            tokenizer=tokenizer,
            mask_prob=cfg["mask_prob"],
            tail_span=cfg.get("tail_span", 3),
            tail_boost=cfg.get("tail_boost", 3.0),
        ),
        drop_last=shuffle,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg["device"] if cfg["device"] != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg["device"] = device
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.startswith("cuda"):
        torch.cuda.manual_seed(cfg["seed"])

    tokenizer = Tokenizer(cfg["tokenizer_model"])
    train_ds = RefinerDataset(
        Path(cfg["train_file"]),
        tokenizer=tokenizer,
        max_length=cfg["max_length"],
        limit=cfg.get("train_limit"),
    )
    val_ds = RefinerDataset(
        Path(cfg["val_file"]),
        tokenizer=tokenizer,
        max_length=cfg["max_length"],
        limit=cfg.get("val_limit"),
    )
    train_loader = make_loader(train_ds, cfg, tokenizer, shuffle=True)
    val_loader = make_loader(val_ds, cfg, tokenizer, shuffle=False)
    train_iter = iter(train_loader)

    model_cfg = RefinerConfig(**cfg["model"])
    model = MaskedRefiner(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and cfg["dtype"] == "float16")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    t0 = time.time()
    for iter_num in range(cfg["max_iters"] + 1):
        if iter_num % cfg["eval_interval"] == 0 or iter_num == cfg["max_iters"]:
            train_loss = estimate_loss(model, train_loader, cfg, device)
            val_loss = estimate_loss(model, val_loader, cfg, device)
            summary = {
                "iter": iter_num,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_sec": round(time.time() - t0, 2),
            }
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            with (out_dir / "log.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_config": cfg["model"],
                        "train_config": cfg,
                        "best_val_loss": best_val,
                    },
                    out_dir / "best.pt",
                )
        if iter_num == cfg["max_iters"]:
            break

        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg["gradient_accumulation_steps"]):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            x = batch["input_ids"].to(device)
            y = batch["targets"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            with autocast_context(device, cfg["dtype"]):
                _, loss = model(x, y, loss_mask)
                loss = loss / cfg["gradient_accumulation_steps"]
            scaler.scale(loss).backward()
        if cfg["grad_clip"]:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

    torch.save(
        {
            "model": model.state_dict(),
            "model_config": cfg["model"],
            "train_config": cfg,
            "best_val_loss": best_val,
        },
        out_dir / "final.pt",
    )

    summary = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_skipped_too_long": train_ds.skipped,
        "val_skipped_too_long": val_ds.skipped,
        "best_val_loss": best_val,
        "config": cfg,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
