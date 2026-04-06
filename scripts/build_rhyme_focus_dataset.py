from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from poetry_lm.text import count_words, split_lines


def write_row(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/processed_rhyme_focus")
    parser.add_argument("--pair-copies", type=int, default=2)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()

    for split in ["train", "val", "test"]:
        in_path = input_dir / f"{split}.jsonl.gz"
        out_path = out_dir / f"{split}.jsonl.gz"
        with (
            gzip.open(in_path, "rt", encoding="utf-8") as in_fh,
            gzip.open(out_path, "wt", encoding="utf-8") as out_fh,
        ):
            for line in in_fh:
                stats[f"{split}_rows_total"] += 1
                row = json.loads(line)
                text = row["text"]
                lines = split_lines(text)
                if len(lines) < 2:
                    stats[f"{split}_rows_skipped_short"] += 1
                    continue
                if row.get("rhyme_scheme") != "AABB":
                    stats[f"{split}_rows_skipped_non_aabb"] += 1
                    continue

                full_row = dict(row)
                full_row["focus_mode"] = "full_poem"
                write_row(out_fh, full_row)
                stats[f"{split}_full_poems"] += 1

                pair_text = "\n".join(lines[:2])
                pair_word_count = count_words(pair_text)
                for pair_idx in range(args.pair_copies):
                    pair_row = dict(row)
                    pair_row["poem_id"] = f"{row['poem_id']}::pair1::{pair_idx}"
                    pair_row["text"] = pair_text
                    pair_row["line_count"] = 2
                    pair_row["word_count"] = pair_word_count
                    pair_row["rhyme_scheme"] = "AA"
                    pair_row["rhyme_confidence"] = 1.0
                    pair_row["focus_mode"] = "first_pair"
                    write_row(out_fh, pair_row)
                    stats[f"{split}_pair_examples"] += 1

        stats[f"{split}_rows_written"] = stats[f"{split}_full_poems"] + stats[f"{split}_pair_examples"]

    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
