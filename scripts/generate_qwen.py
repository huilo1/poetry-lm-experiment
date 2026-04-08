from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from poetry_lm.qwen_sft import extract_generated_lines, render_prompt_prefix


def build_quantization_config(load_in_4bit: bool, bf16: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    quantization_config = build_quantization_config(args.load_in_4bit, args.bf16)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto" if args.device != "cpu" else None,
        quantization_config=quantization_config,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    prompt_text = render_prompt_prefix(args.prompt)
    model_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    generated = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = generated[0][model_inputs["input_ids"].shape[1] :]
    continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    lines = extract_generated_lines(args.prompt, continuation)
    text = "\n".join(lines)
    print(text)
    metrics_path = Path(args.adapter_dir) / "generate.json"
    metrics_path.write_text(
        json.dumps({"prompt": args.prompt, "output": text}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
