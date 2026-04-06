from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_memmap(dataset_dir: Path, split: str) -> np.memmap:
    meta = json.loads((dataset_dir / "meta.json").read_text(encoding="utf-8"))
    dtype = np.uint16 if meta["dtype"] == "uint16" else np.uint32
    return np.memmap(dataset_dir / f"{split}.bin", dtype=dtype, mode="r")


def get_batch(
    data: np.memmap, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
    )
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg: dict) -> dict[str, float]:
    model.eval()
    out = {}
    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg["eval_steps"])
        for k in range(cfg["eval_steps"]):
            x, y = get_batch(data, cfg["block_size"], cfg["batch_size"], cfg["device"])
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out
