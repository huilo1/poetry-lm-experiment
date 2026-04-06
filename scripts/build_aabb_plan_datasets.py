from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from poetry_lm.planning import ending_words_from_lines
from poetry_lm.text import count_words, split_lines
from poetry_lm.tokenizer import (
    PLANNER_MODE_AABB,
    STRUCTURED_MODE_AABB_PLAN,
    planner_window_to_training_text,
    structured_window_to_training_text_with_plan,
)


def planner_row_from_base(row: dict, lines: list[str], plan_endings: dict[str, str]) -> dict:
    return {
        **row,
        "focus_mode": "ending_planner",
        "prompt_mode": PLANNER_MODE_AABB,
        "plan_endings": plan_endings,
        "formatted_text": planner_window_to_training_text(lines, plan_endings),
        "word_count": count_words("\n".join(lines)),
    }


def generator_row_from_base(row: dict, lines: list[str], plan_endings: dict[str, str]) -> dict:
    return {
        **row,
        "focus_mode": "generator_with_plan",
        "prompt_mode": STRUCTURED_MODE_AABB_PLAN,
        "plan_endings": plan_endings,
        "formatted_text": structured_window_to_training_text_with_plan(lines, plan_endings),
        "word_count": count_words("\n".join(lines)),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed_aabb8_qf2")
    parser.add_argument("--planner-out-dir", default="data/processed_aabb8_plan")
    parser.add_argument("--generator-out-dir", default="data/processed_aabb8_planned")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    planner_out_dir = Path(args.planner_out_dir)
    generator_out_dir = Path(args.generator_out_dir)
    planner_out_dir.mkdir(parents=True, exist_ok=True)
    generator_out_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()

    for split in ["train", "val", "test"]:
        planner_rows: list[dict] = []
        generator_rows: list[dict] = []
        planner_preview: list[str] = []
        generator_preview: list[str] = []
        input_path = input_dir / f"{split}.jsonl.gz"

        with gzip.open(input_path, "rt", encoding="utf-8") as fh:
            for idx, line in enumerate(tqdm(fh, desc=f"build_plan_{split}")):
                if args.max_rows is not None and idx >= args.max_rows:
                    break
                row = json.loads(line)
                lines = split_lines(row["text"])
                if len(lines) != 8:
                    stats[f"{split}_wrong_line_count"] += 1
                    continue
                plan_endings = ending_words_from_lines(lines)
                if not all(plan_endings.values()):
                    stats[f"{split}_missing_plan_word"] += 1
                    continue
                planner_rows.append(planner_row_from_base(row, lines, plan_endings))
                generator_rows.append(generator_row_from_base(row, lines, plan_endings))
                stats[f"{split}_kept"] += 1

                if len(planner_preview) < 6:
                    planner_preview.append(planner_rows[-1]["formatted_text"])
                if len(generator_preview) < 4:
                    generator_preview.append(generator_rows[-1]["formatted_text"])

        write_jsonl(planner_out_dir / f"{split}.jsonl.gz", planner_rows)
        write_jsonl(generator_out_dir / f"{split}.jsonl.gz", generator_rows)
        (planner_out_dir / f"{split}.preview.txt").write_text("\n\n".join(planner_preview), encoding="utf-8")
        (generator_out_dir / f"{split}.preview.txt").write_text("\n\n".join(generator_preview), encoding="utf-8")

    stats_payload = json.dumps(stats, ensure_ascii=False, indent=2)
    (planner_out_dir / "stats.json").write_text(stats_payload, encoding="utf-8")
    (generator_out_dir / "stats.json").write_text(stats_payload, encoding="utf-8")


if __name__ == "__main__":
    main()
