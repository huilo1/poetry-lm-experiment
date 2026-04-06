from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from itertools import islice
from pathlib import Path

from tqdm import tqdm

from poetry_lm.dataset import poem_is_good
from poetry_lm.rhyme import (
    detect_eight_line_abab_abab,
    detect_eight_line_abab_abab_from_tails,
    lines_rhyme_tails_from_accented,
)
from poetry_lm.stress import accent_last_words
from poetry_lm.text import count_words, split_lines, stable_normalize_for_hash
from poetry_lm.tokenizer import STRUCTURED_SCHEME_ABAB, structured_window_to_training_text


def build_window_record(row: dict, lines: list[str], start_idx: int, confidence: float) -> dict:
    text = "\n".join(lines)
    content_hash = hashlib.sha1(stable_normalize_for_hash(lines).encode("utf-8")).hexdigest()
    return {
        "source": row.get("source", "stihi_ru"),
        "poem_id": f"{row['poem_id']}::window8::{start_idx}",
        "parent_poem_id": row["poem_id"],
        "content_hash": content_hash,
        "author": row.get("author"),
        "title": row.get("title"),
        "genre": row.get("genre"),
        "topic": row.get("topic"),
        "text": text,
        "formatted_text": structured_window_to_training_text(lines, scheme=STRUCTURED_SCHEME_ABAB),
        "prompt": lines[0],
        "line_count": 8,
        "word_count": count_words(text),
        "rhyme_scheme": STRUCTURED_SCHEME_ABAB,
        "rhyme_confidence": confidence,
        "window_start": start_idx,
    }


def write_row(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def flush_pending(split: str, pending: list[tuple[dict, list[str], int]], out_fh, stats: Counter, seen: set[str]) -> None:
    if not pending:
        return

    flat_lines = [line for _, window, _ in pending for line in window]
    accented = accent_last_words(flat_lines)

    for batch_idx, (row, window, start_idx) in enumerate(pending):
        start = batch_idx * 8
        tails = lines_rhyme_tails_from_accented(window, accented[start : start + 8])
        stressed = detect_eight_line_abab_abab_from_tails(tails)
        if stressed.scheme != STRUCTURED_SCHEME_ABAB:
            stats[f"{split}_windows_rejected_stress"] += 1
            continue
        record = build_window_record(row, window, start_idx, stressed.confidence)
        if record["content_hash"] in seen:
            stats[f"{split}_windows_deduped"] += 1
            continue
        seen.add(record["content_hash"])
        write_row(out_fh, record)
        stats[f"{split}_windows_kept"] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/processed_abab8")
    parser.add_argument("--max-poems", type=int, default=None)
    parser.add_argument("--all-windows", action="store_true")
    parser.add_argument("--stress-batch-size", type=int, default=256)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    seen: set[str] = set()

    for split in ["train", "val", "test"]:
        in_path = input_dir / f"{split}.jsonl.gz"
        out_path = out_dir / f"{split}.jsonl.gz"
        preview: list[str] = []
        with (
            gzip.open(in_path, "rt", encoding="utf-8") as in_fh,
            gzip.open(out_path, "wt", encoding="utf-8") as out_fh,
        ):
            iterator = in_fh if args.max_poems is None else islice(in_fh, args.max_poems)
            pending: list[tuple[dict, list[str], int]] = []
            for line in tqdm(iterator, desc=f"build_{split}"):
                row = json.loads(line)
                if row.get("focus_mode") == "first_pair":
                    stats[f"{split}_rows_skipped_pair_mode"] += 1
                    continue
                if row.get("rhyme_scheme") != "ABAB":
                    stats[f"{split}_rows_skipped_non_abab"] += 1
                    continue
                if row.get("line_count", 0) < 8:
                    stats[f"{split}_poems_too_short"] += 1
                    continue
                poem_lines = split_lines(row["text"])
                window_starts = range(len(poem_lines) - 7) if args.all_windows else [0]
                for start_idx in window_starts:
                    stats[f"{split}_windows_total"] += 1
                    window = poem_lines[start_idx : start_idx + 8]
                    if not poem_is_good(window):
                        stats[f"{split}_windows_bad_shape"] += 1
                        continue
                    heuristic = detect_eight_line_abab_abab(window, use_stress=False)
                    if heuristic.scheme != STRUCTURED_SCHEME_ABAB:
                        stats[f"{split}_windows_rejected_heuristic"] += 1
                        continue
                    pending.append((row, window, start_idx))
                    if len(pending) >= args.stress_batch_size:
                        flush_pending(split, pending, out_fh, stats, seen)
                        pending.clear()
                    if len(preview) < 12:
                        preview.extend(window)
                        preview.append("")
            flush_pending(split, pending, out_fh, stats, seen)
        (out_dir / f"{split}.preview.txt").write_text("\n".join(preview), encoding="utf-8")

    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
