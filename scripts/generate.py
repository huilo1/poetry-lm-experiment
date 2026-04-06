from __future__ import annotations

import argparse
import json
from pathlib import Path

from poetry_lm.inference import generate_text, load_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/tiny_cpu/best.pt")
    parser.add_argument("--tokenizer-model", default="artifacts/tokenizer/poetry.model")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    bundle = load_bundle(Path(args.checkpoint), args.tokenizer_model, device=args.device)
    text = generate_text(
        bundle=bundle,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)
    metrics_path = Path(args.checkpoint).with_suffix(".generate.json")
    metrics_path.write_text(
        json.dumps({"prompt": args.prompt, "output": text}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
