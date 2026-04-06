from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from itertools import islice
from pathlib import Path

from tqdm import tqdm

from poetry_lm.dataset import (
    append_jsonl_gz_line,
    build_record,
    deterministic_split_key,
    iter_stihi_records,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/stihi_ru.jsonl.zst")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dedup = set()
    stats = Counter()
    preview_lines: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    train_path = out_dir / "train.jsonl.gz"
    val_path = out_dir / "val.jsonl.gz"
    test_path = out_dir / "test.jsonl.gz"

    iterator = iter_stihi_records(input_path)
    if args.limit is not None:
        iterator = islice(iterator, args.limit)

    with (
        gzip.open(train_path, "wt", encoding="utf-8") as train_fh,
        gzip.open(val_path, "wt", encoding="utf-8") as val_fh,
        gzip.open(test_path, "wt", encoding="utf-8") as test_fh,
    ):
        for row in tqdm(iterator, desc="build_dataset"):
            stats["rows_total"] += 1
            record = build_record(row)
            if record is None:
                stats["rows_filtered"] += 1
                continue
            if record.content_hash in dedup:
                stats["rows_deduped"] += 1
                continue
            dedup.add(record.content_hash)
            bucket = deterministic_split_key(record.author, record.poem_id)
            if bucket < 0.9:
                split = "train"
                append_jsonl_gz_line(train_fh, record)
            elif bucket < 0.97:
                split = "val"
                append_jsonl_gz_line(val_fh, record)
            else:
                split = "test"
                append_jsonl_gz_line(test_fh, record)
            stats["rows_kept"] += 1
            stats[f"scheme_{record.rhyme_scheme}"] += 1
            stats[f"{split}_size"] += 1
            if len(preview_lines[split]) < 6:
                preview_lines[split].append(record.text)
                preview_lines[split].append("")

    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for split, lines in preview_lines.items():
        preview_path = out_dir / f"{split}.preview.txt"
        preview_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
