from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from poetry_lm.qwen_sft import extract_generated_lines, render_prompt_prefix
from poetry_lm.rhyme import detect_eight_line_aabb_aabb, rhymes


def build_quantization_config(load_in_4bit: bool, bf16: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=100)
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

    total = 0
    exact_8 = 0
    scheme_ok = 0
    second_line_rhyme = 0
    preview = []

    with gzip.open(args.test_file, "rt", encoding="utf-8") as fh:
        for line in tqdm(fh, total=args.limit, desc="evaluate_qwen_8line"):
            if total >= args.limit:
                break
            row = json.loads(line)
            prompt = row["prompt"]
            prompt_text = render_prompt_prefix(prompt)
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
            lines = extract_generated_lines(prompt, continuation)
            if len(lines) >= 2 and rhymes(lines[0], lines[1], use_stress=True):
                second_line_rhyme += 1
            if len(lines) == 8:
                exact_8 += 1
            if detect_eight_line_aabb_aabb(lines[:8], use_stress=True).scheme == "AABB_CCDD":
                scheme_ok += 1
            preview.append({"prompt": prompt, "generated": "\n".join(lines[:8])})
            total += 1

    report = {
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "samples": total,
        "exact_8_lines_rate": exact_8 / max(total, 1),
        "second_line_rhyme_rate": second_line_rhyme / max(total, 1),
        "aabb_ccdd_rate": scheme_ok / max(total, 1),
        "preview": preview[:5],
    }
    out_path = Path(args.adapter_dir) / "eval8.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
