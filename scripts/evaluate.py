from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import torch
from tqdm import tqdm

from poetry_lm.model import GPT, GPTConfig
from poetry_lm.rhyme import rhymes
from poetry_lm.tokenizer import Tokenizer


def load_model(checkpoint_path: Path, device: str) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/tiny_cpu/best.pt")
    parser.add_argument("--tokenizer-model", default="artifacts/tokenizer/poetry.model")
    parser.add_argument("--test-file", default="data/processed/test.jsonl.gz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer_model)
    model = load_model(Path(args.checkpoint), args.device)

    total = 0
    rhyme_ok = 0
    samples = []

    with gzip.open(args.test_file, "rt", encoding="utf-8") as fh:
        for line in tqdm(fh, total=args.limit, desc="evaluate"):
            if total >= args.limit:
                break
            row = json.loads(line)
            poem_lines = [ln for ln in row["text"].splitlines() if ln.strip()]
            if len(poem_lines) < 2:
                continue
            prompt = poem_lines[0]
            x = torch.tensor([tokenizer.encode_prompt(prompt)], dtype=torch.long, device=args.device)
            out = model.generate(
                x,
                max_new_tokens=64,
                temperature=0.9,
                top_k=50,
                eos_id=tokenizer.eos_id,
            )
            generated = tokenizer.decode(out[0].tolist()).splitlines()
            if len(generated) >= 2 and rhymes(generated[0], generated[1]):
                rhyme_ok += 1
            samples.append({"prompt": prompt, "generated": "\n".join(generated[:4])})
            total += 1

    report = {
        "samples": total,
        "second_line_rhyme_rate": rhyme_ok / max(total, 1),
        "preview": samples[:5],
    }
    out_path = Path(args.checkpoint).with_suffix(".eval.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
