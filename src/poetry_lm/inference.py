from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import torch

from poetry_lm.model import GPT, GPTConfig
from poetry_lm.tokenizer import PLANNER_MODE_AABB, STRUCTURED_MODE_AABB_PLAN, Tokenizer


@dataclass
class LoadedBundle:
    checkpoint_path: Path
    tokenizer_model: Path
    device: str
    prompt_mode: str
    model: GPT
    tokenizer: Tokenizer
    checkpoint_mtime_ns: int


_CACHE: dict[tuple[str, str, str], LoadedBundle] = {}
_CACHE_LOCK = Lock()


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_bundle(
    checkpoint_path: str | Path,
    tokenizer_model: str | Path,
    device: str = "auto",
) -> LoadedBundle:
    resolved_device = resolve_device(device)
    checkpoint = Path(checkpoint_path).resolve()
    tokenizer_path = Path(tokenizer_model).resolve()
    cache_key = (str(checkpoint), str(tokenizer_path), resolved_device)
    checkpoint_mtime_ns = checkpoint.stat().st_mtime_ns

    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached and cached.checkpoint_mtime_ns == checkpoint_mtime_ns:
            return cached

    ckpt = torch.load(checkpoint, map_location=resolved_device)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(resolved_device)
    model.eval()

    bundle = LoadedBundle(
        checkpoint_path=checkpoint,
        tokenizer_model=tokenizer_path,
        device=resolved_device,
        prompt_mode=ckpt.get("train_config", {}).get("prompt_mode", "legacy"),
        model=model,
        tokenizer=Tokenizer(tokenizer_path),
        checkpoint_mtime_ns=checkpoint_mtime_ns,
    )

    with _CACHE_LOCK:
        _CACHE[cache_key] = bundle

    return bundle


@torch.no_grad()
def generate_text(
    bundle: LoadedBundle,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_k: int = 50,
    plan_endings: dict[str, str] | None = None,
) -> str:
    if bundle.prompt_mode == STRUCTURED_MODE_AABB_PLAN and plan_endings is None:
        raise ValueError("planned generation mode requires plan_endings")
    prompt_ids = bundle.tokenizer.encode_prompt(prompt, mode=bundle.prompt_mode, plan_endings=plan_endings)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=bundle.device)
    out = bundle.model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=bundle.tokenizer.eos_id,
    )
    text = bundle.tokenizer.decode(out[0].tolist(), mode=bundle.prompt_mode)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


@torch.no_grad()
def generate_plan(
    bundle: LoadedBundle,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    top_k: int = 20,
) -> dict[str, str]:
    if bundle.prompt_mode != PLANNER_MODE_AABB:
        raise ValueError(f"bundle prompt_mode={bundle.prompt_mode} is not a planner mode")
    prompt_ids = bundle.tokenizer.encode_prompt(prompt, mode=bundle.prompt_mode)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=bundle.device)
    out = bundle.model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=bundle.tokenizer.eos_id,
    )
    return bundle.tokenizer.decode_plan(out[0].tolist())


@torch.no_grad()
def generate_text_with_planner(
    planner_bundle: LoadedBundle,
    generator_bundle: LoadedBundle,
    prompt: str,
    max_plan_tokens: int = 32,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_k: int = 50,
    planner_temperature: float = 0.8,
    planner_top_k: int = 20,
) -> tuple[dict[str, str], str]:
    plan = generate_plan(
        bundle=planner_bundle,
        prompt=prompt,
        max_new_tokens=max_plan_tokens,
        temperature=planner_temperature,
        top_k=planner_top_k,
    )
    text = generate_text(
        bundle=generator_bundle,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        plan_endings=plan,
    )
    return plan, text
