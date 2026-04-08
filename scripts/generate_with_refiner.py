from __future__ import annotations

import argparse
from pathlib import Path

from poetry_lm.inference import generate_text, load_bundle
from poetry_lm.refiner import load_refiner, refine_draft_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--baseline-tokenizer", required=True)
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--refiner-tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--refiner-steps", type=int, default=8)
    parser.add_argument("--refiner-temperature", type=float, default=0.8)
    parser.add_argument("--refiner-top-k", type=int, default=32)
    args = parser.parse_args()

    baseline = load_bundle(Path(args.baseline_checkpoint), args.baseline_tokenizer, device=args.device)
    refiner, refiner_tokenizer, _ = load_refiner(
        Path(args.refiner_checkpoint),
        args.refiner_tokenizer,
        device=baseline.device,
    )
    draft = generate_text(
        baseline,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    refined = refine_draft_text(
        model=refiner,
        tokenizer=refiner_tokenizer,
        draft_text=draft,
        device=baseline.device,
        steps=args.refiner_steps,
        temperature=args.refiner_temperature,
        top_k=args.refiner_top_k,
    )
    print("=== draft ===")
    print(draft)
    print("\n=== refined ===")
    print(refined)


if __name__ == "__main__":
    main()
