from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import torch
from tqdm import tqdm

from poetry_lm.model import GPT, GPTConfig
from poetry_lm.rhyme import detect_eight_line_aabb_aabb, detect_eight_line_abab_abab, rhymes
from poetry_lm.tokenizer import STRUCTURED_MODE, STRUCTURED_MODE_ABAB, Tokenizer


def load_model(checkpoint_path: Path, device: str) -> tuple[GPT, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer_model)
    model, ckpt = load_model(Path(args.checkpoint), args.device)
    prompt_mode = ckpt.get("train_config", {}).get("prompt_mode", STRUCTURED_MODE)
    if prompt_mode == STRUCTURED_MODE_ABAB:
        target_scheme = "ABAB_ABAB"
        detect_scheme = detect_eight_line_abab_abab
        metric_key = "abab_abab_rate"
    else:
        target_scheme = "AABB_CCDD"
        detect_scheme = detect_eight_line_aabb_aabb
        metric_key = "aabb_ccdd_rate"

    total = 0
    exact_8 = 0
    scheme_ok = 0
    second_line_rhyme = 0
    preview = []

    with gzip.open(args.test_file, "rt", encoding="utf-8") as fh:
        for line in tqdm(fh, total=args.limit, desc="evaluate_8line"):
            if total >= args.limit:
                break
            row = json.loads(line)
            prompt = row.get("prompt") or row["text"].splitlines()[0]
            x = torch.tensor(
                [tokenizer.encode_prompt(prompt, mode=prompt_mode)],
                dtype=torch.long,
                device=args.device,
            )
            out = model.generate(
                x,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_id=tokenizer.eos_id,
            )
            generated = tokenizer.decode(out[0].tolist(), mode=prompt_mode)
            lines = [ln.strip() for ln in generated.splitlines() if ln.strip()]
            if len(lines) >= 2 and rhymes(lines[0], lines[1], use_stress=True):
                second_line_rhyme += 1
            if len(lines) == 8:
                exact_8 += 1
            if detect_scheme(lines[:8], use_stress=True).scheme == target_scheme:
                scheme_ok += 1
            preview.append({"prompt": prompt, "generated": "\n".join(lines[:8])})
            total += 1

    report = {
        "samples": total,
        "exact_8_lines_rate": exact_8 / max(total, 1),
        "second_line_rhyme_rate": second_line_rhyme / max(total, 1),
        "preview": preview[:5],
    }
    report[metric_key] = scheme_ok / max(total, 1)
    out_path = Path(args.checkpoint).with_suffix(".eval8.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
