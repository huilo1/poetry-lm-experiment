from __future__ import annotations

import argparse
import json
from pathlib import Path

from poetry_lm.inference import generate_text_with_planner, load_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-checkpoint", required=True)
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-plan-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--planner-temperature", type=float, default=0.8)
    parser.add_argument("--planner-top-k", type=int, default=20)
    args = parser.parse_args()

    planner = load_bundle(Path(args.planner_checkpoint), args.tokenizer_model, device=args.device)
    generator = load_bundle(Path(args.generator_checkpoint), args.tokenizer_model, device=args.device)
    plan, text = generate_text_with_planner(
        planner_bundle=planner,
        generator_bundle=generator,
        prompt=args.prompt,
        max_plan_tokens=args.max_plan_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        planner_temperature=args.planner_temperature,
        planner_top_k=args.planner_top_k,
    )
    payload = {"prompt": args.prompt, "plan": plan, "output": text}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
