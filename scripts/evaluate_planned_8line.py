from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from tqdm import tqdm

from poetry_lm.inference import generate_text_with_planner, load_bundle
from poetry_lm.rhyme import detect_eight_line_aabb_aabb, rhymes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-checkpoint", required=True)
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-plan-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--planner-temperature", type=float, default=0.8)
    parser.add_argument("--planner-top-k", type=int, default=20)
    args = parser.parse_args()

    planner = load_bundle(Path(args.planner_checkpoint), args.tokenizer_model, device=args.device)
    generator = load_bundle(Path(args.generator_checkpoint), args.tokenizer_model, device=args.device)

    total = 0
    exact_8 = 0
    scheme_ok = 0
    second_line_rhyme = 0
    preview = []

    with gzip.open(args.test_file, "rt", encoding="utf-8") as fh:
        for line in tqdm(fh, total=args.limit, desc="evaluate_planned_8line"):
            if total >= args.limit:
                break
            row = json.loads(line)
            prompt = row.get("prompt") or row["text"].splitlines()[0]
            plan, generated = generate_text_with_planner(
                planner_bundle=planner,
                generator_bundle=generator,
                prompt=prompt,
                max_plan_tokens=args.max_plan_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                planner_temperature=args.planner_temperature,
                planner_top_k=args.planner_top_k,
            )
            lines = [ln.strip() for ln in generated.splitlines() if ln.strip()]
            if len(lines) >= 2 and rhymes(lines[0], lines[1], use_stress=True):
                second_line_rhyme += 1
            if len(lines) == 8:
                exact_8 += 1
            if detect_eight_line_aabb_aabb(lines[:8], use_stress=True).scheme == "AABB_CCDD":
                scheme_ok += 1
            preview.append({"prompt": prompt, "plan": plan, "generated": "\n".join(lines[:8])})
            total += 1

    report = {
        "samples": total,
        "exact_8_lines_rate": exact_8 / max(total, 1),
        "second_line_rhyme_rate": second_line_rhyme / max(total, 1),
        "aabb_ccdd_rate": scheme_ok / max(total, 1),
        "preview": preview[:5],
    }
    out_path = Path(args.generator_checkpoint).with_suffix(".planned.eval8.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
