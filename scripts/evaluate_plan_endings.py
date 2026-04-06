from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from tqdm import tqdm

from poetry_lm.inference import generate_plan, load_bundle
from poetry_lm.planning import ending_words_from_text
from poetry_lm.rhyme import rhyme_tail_from_word
from poetry_lm.tokenizer import PLAN_TAGS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    bundle = load_bundle(Path(args.checkpoint), args.tokenizer_model, device=args.device)

    total = 0
    exact = 0
    rhyme_tail_match = 0
    preview = []

    with gzip.open(args.test_file, "rt", encoding="utf-8") as fh:
        for line in tqdm(fh, total=args.limit, desc="evaluate_plan_endings"):
            if total >= args.limit:
                break
            row = json.loads(line)
            prompt = row.get("prompt") or row["text"].splitlines()[0]
            gold = row.get("plan_endings") or ending_words_from_text(row["text"])
            pred = generate_plan(
                bundle=bundle,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            for tag in PLAN_TAGS:
                if pred.get(tag, "") == gold.get(tag, ""):
                    exact += 1
                if rhyme_tail_from_word(pred.get(tag, "")) == rhyme_tail_from_word(gold.get(tag, "")):
                    rhyme_tail_match += 1
            preview.append({"prompt": prompt, "gold": gold, "pred": pred})
            total += 1

    denom = max(total * len(PLAN_TAGS), 1)
    report = {
        "samples": total,
        "ending_exact_match_rate": exact / denom,
        "ending_rhyme_tail_match_rate": rhyme_tail_match / denom,
        "preview": preview[:5],
    }
    out_path = Path(args.checkpoint).with_suffix(".plan.eval.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
