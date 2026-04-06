from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from poetry_lm.tokenizer import sample_to_training_text, train_sentencepiece


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=["data/processed/train.jsonl.gz"])
    parser.add_argument("--tmp-text", default="data/processed/tokenizer_input.txt")
    parser.add_argument("--model-prefix", default="artifacts/tokenizer/poetry")
    parser.add_argument("--vocab-size", type=int, default=16000)
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.input]
    tmp_path = Path(args.tmp_text)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    with tmp_path.open("w", encoding="utf-8") as out:
        for input_path in input_paths:
            with gzip.open(input_path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    out.write(sample_to_training_text(row))
                    out.write("\n")

    train_sentencepiece(tmp_path, Path(args.model_prefix), args.vocab_size)


if __name__ == "__main__":
    main()
