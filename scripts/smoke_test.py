from __future__ import annotations

import argparse
import json
from pathlib import Path

from poetry_lm.rhyme import detect_eight_line_aabb_aabb, detect_quatrain_scheme, rhymes
from poetry_lm.tokenizer import STRUCTURED_MODE, Tokenizer, structured_window_to_training_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-stats", default="data/processed/stats.json")
    args = parser.parse_args()

    assert rhymes("любовь", "кровь")
    assert rhymes("дорога", "тревога")
    scheme = detect_quatrain_scheme(["мама", "рама", "свет", "ответ"])
    assert scheme.scheme == "AABB"
    structured = detect_eight_line_aabb_aabb(
        ["мама", "рама", "свет", "ответ", "дама", "реклама", "след", "рассвет"]
    )
    assert structured.scheme == "AABB_CCDD"
    assert "<SCHEME_AABB_CCDD>" in structured_window_to_training_text(
        ["один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь"]
    )
    tokenizer = Tokenizer("artifacts/tokenizer/poetry.model")
    assert tokenizer.format_prompt("первая строка", mode=STRUCTURED_MODE).startswith("<PROMPT>")

    stats = json.loads(Path(args.dataset_stats).read_text(encoding="utf-8"))
    assert stats["rows_kept"] > 0
    assert stats["train_size"] > 0
    print("smoke_test_ok")


if __name__ == "__main__":
    main()
