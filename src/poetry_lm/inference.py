from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import torch

from poetry_lm.model import GPT, GPTConfig
from poetry_lm.tokenizer import PLANNER_MODE_AABB, STRUCTURED_MODE_AABB_PLAN, Tokenizer
from poetry_lm.gigachat_sft import (
    extract_generated_lines,
    prepare_gigachat_local_model_dir,
    render_prompt_prefix,
)


@dataclass
class LoadedBundle:
    checkpoint_path: Path
    tokenizer_model: Path
    device: str
    prompt_mode: str
    model: GPT
    tokenizer: Tokenizer
    checkpoint_mtime_ns: int


@dataclass
class LoadedGigaChatBundle:
    adapter_dir: Path
    base_model: str
    device: str
    load_in_4bit: bool
    bf16: bool
    model: object
    tokenizer: object
    adapter_mtime_ns: int


_CACHE: dict[tuple[str, str, str], LoadedBundle] = {}
_GIGACHAT_CACHE: dict[tuple[str, str, str, bool, bool], LoadedGigaChatBundle] = {}
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


def _build_quantization_config(load_in_4bit: bool, bf16: bool):
    if not load_in_4bit:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
    )


def load_gigachat_bundle(
    adapter_dir: str | Path,
    base_model: str,
    device: str = "auto",
    load_in_4bit: bool = True,
    bf16: bool = True,
) -> LoadedGigaChatBundle:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = resolve_device(device)
    adapter_path = Path(adapter_dir).resolve()
    adapter_model_path = adapter_path / "adapter_model.safetensors"
    adapter_mtime_ns = adapter_model_path.stat().st_mtime_ns
    cache_key = (str(adapter_path), base_model, resolved_device, load_in_4bit, bf16)

    with _CACHE_LOCK:
        cached = _GIGACHAT_CACHE.get(cache_key)
        if cached and cached.adapter_mtime_ns == adapter_mtime_ns:
            return cached

    local_base_model = prepare_gigachat_local_model_dir(base_model)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = _build_quantization_config(load_in_4bit=load_in_4bit, bf16=bf16)
    model = AutoModelForCausalLM.from_pretrained(
        local_base_model,
        dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto" if resolved_device != "cpu" else None,
        quantization_config=quantization_config,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    bundle = LoadedGigaChatBundle(
        adapter_dir=adapter_path,
        base_model=base_model,
        device=resolved_device,
        load_in_4bit=load_in_4bit,
        bf16=bf16,
        model=model,
        tokenizer=tokenizer,
        adapter_mtime_ns=adapter_mtime_ns,
    )

    with _CACHE_LOCK:
        _GIGACHAT_CACHE[cache_key] = bundle

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


@torch.no_grad()
def generate_gigachat_text(
    bundle: LoadedGigaChatBundle,
    prompt: str,
    max_new_tokens: int = 160,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    prompt_text = render_prompt_prefix(prompt)
    model_inputs = bundle.tokenizer(prompt_text, return_tensors="pt").to(bundle.model.device)
    generated = bundle.model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
        pad_token_id=bundle.tokenizer.pad_token_id,
        eos_token_id=bundle.tokenizer.eos_token_id,
    )
    generated_ids = generated[0][model_inputs["input_ids"].shape[1] :]
    continuation = bundle.tokenizer.decode(generated_ids, skip_special_tokens=True)
    lines = extract_generated_lines(prompt, continuation)
    return "\n".join(lines)
