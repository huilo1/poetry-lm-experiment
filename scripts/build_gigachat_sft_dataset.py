from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from poetry_lm.gigachat_sft import format_dataset_row


def convert_split(input_path: Path, output_path: Path) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"rows_seen": 0, "rows_written": 0, "rows_skipped": 0}
    with gzip.open(input_path, "rt", encoding="utf-8") as src, gzip.open(
        output_path, "wt", encoding="utf-8"
    ) as dst:
        for line in src:
            stats["rows_seen"] += 1
            row = json.loads(line)
            try:
                formatted = format_dataset_row(row)
            except ValueError:
                stats["rows_skipped"] += 1
                continue
            dst.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            stats["rows_written"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed_aabb8_qf2")
    parser.add_argument("--out-dir", default="data/gigachat_aabb_qf2_sft")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    overall = {}
    for split in ("train", "val", "test"):
        input_path = input_dir / f"{split}.jsonl.gz"
        output_path = out_dir / f"{split}.jsonl.gz"
        overall[split] = convert_split(input_path, output_path)
    (out_dir / "stats.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
