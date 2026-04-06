from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from poetry_lm.model import GPT, GPTConfig, learning_rate_for_iter
from poetry_lm.training import estimate_loss, get_batch, get_memmap, load_config


def autocast_context(device: str, dtype_name: str):
    if device == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    return torch.autocast(device_type="cuda", dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_cpu.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg["device"] if cfg["device"] != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg["device"] = device
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg["seed"])
    if device.startswith("cuda"):
        torch.cuda.manual_seed(cfg["seed"])

    dataset_dir = Path(cfg["dataset_dir"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train_data = get_memmap(dataset_dir, "train")
    val_data = get_memmap(dataset_dir, "val")

    model_cfg = GPTConfig(**cfg["model"])
    model = GPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and cfg["dtype"] == "float16")

    start_iter = 0
    best_val = float("inf")

    if args.resume and args.init_from:
        raise ValueError("use either --resume or --init-from, not both")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter_num"] + 1
        best_val = ckpt["best_val_loss"]
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ckpt["model"])

    if cfg.get("compile") and device.startswith("cuda"):
        model = torch.compile(model)

    t0 = time.time()
    for iter_num in range(start_iter, cfg["max_iters"] + 1):
        lr = learning_rate_for_iter(
            iter_num, cfg["learning_rate"], cfg["warmup_iters"], cfg["max_iters"]
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        if iter_num % cfg["eval_interval"] == 0 or iter_num == cfg["max_iters"]:
            losses = estimate_loss(model, train_data, val_data, cfg)
            summary = {
                "iter": iter_num,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "lr": lr,
                "elapsed_sec": round(time.time() - t0, 2),
            }
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            (out_dir / "log.jsonl").open("a", encoding="utf-8").write(
                json.dumps(summary, ensure_ascii=False) + "\n"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt_path = out_dir / "best.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": cfg["model"],
                        "iter_num": iter_num,
                        "best_val_loss": best_val,
                        "train_config": cfg,
                    },
                    ckpt_path,
                )

        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(cfg["gradient_accumulation_steps"]):
            x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
            with autocast_context(device, cfg["dtype"]):
                _, loss = model(x, y)
                loss = loss / cfg["gradient_accumulation_steps"]
            scaler.scale(loss).backward()
        if cfg["grad_clip"]:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

    final_path = out_dir / "final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": cfg["model"],
            "train_config": cfg,
            "best_val_loss": best_val,
        },
        final_path,
    )


if __name__ == "__main__":
    main()
