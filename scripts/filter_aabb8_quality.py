from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from tqdm import tqdm

from poetry_lm.quality import score_window
from poetry_lm.rhyme import lines_rhyme_tails_from_accented
from poetry_lm.stress import accent_last_words
from poetry_lm.text import split_lines


def write_row(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_pending(
    pending: list[tuple[dict, list[str]]],
    out_fh,
    split: str,
    stats: Counter,
    reject_examples: dict[str, list[dict]],
    preview_kept: list[str],
    preview_rejected: list[str],
    min_score: float,
    default_scheme: str,
) -> None:
    if not pending:
        return

    flat_lines = [line for _, lines in pending for line in lines]
    accented = accent_last_words(flat_lines)

    for batch_idx, (row, lines) in enumerate(pending):
        start = batch_idx * 8
        tails = lines_rhyme_tails_from_accented(lines, accented[start : start + 8])
        quality = score_window(lines, tails=tails, scheme=row.get("rhyme_scheme", default_scheme))
        stats[f"{split}_score_sum_x1000"] += int(quality.score * 1000)

        row["quality_score"] = quality.score
        row["quality_hard_reasons"] = quality.hard_reasons
        row["quality_soft_reasons"] = quality.soft_reasons
        row["quality_features"] = quality.features

        keep = quality.keep and quality.score >= min_score
        if keep:
            write_row(out_fh, row)
            stats[f"{split}_rows_kept"] += 1
            if len(preview_kept) < 20:
                preview_kept.append(
                    f"[score={quality.score}] {row['poem_id']}\n{row['text']}\n"
                )
            continue

        stats[f"{split}_rows_rejected"] += 1
        for reason in quality.hard_reasons:
            stats[f"{split}_hard_{reason}"] += 1
        for reason in quality.soft_reasons:
            stats[f"{split}_soft_{reason}"] += 1
        if quality.score < min_score:
            stats[f"{split}_below_score_threshold"] += 1
        if len(preview_rejected) < 20:
            reasons = ",".join(quality.hard_reasons + quality.soft_reasons) or "score_only"
            preview_rejected.append(
                f"[score={quality.score}] {row['poem_id']} [{reasons}]\n{row['text']}\n"
            )
        for reason in (quality.hard_reasons or ["score_only"])[:3]:
            if len(reject_examples[reason]) < 3:
                reject_examples[reason].append(
                    {
                        "split": split,
                        "poem_id": row["poem_id"],
                        "score": quality.score,
                        "text": row["text"],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed_aabb8")
    parser.add_argument("--out-dir", default="data/processed_aabb8_qf2")
    parser.add_argument("--min-score", type=float, default=0.95)
    parser.add_argument("--max-poems", type=int, default=None)
    parser.add_argument("--stress-batch-size", type=int, default=256)
    parser.add_argument("--default-scheme", default="AABB_CCDD")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    reject_examples: dict[str, list[dict]] = defaultdict(list)

    for split in ["train", "val", "test"]:
        in_path = input_dir / f"{split}.jsonl.gz"
        out_path = out_dir / f"{split}.jsonl.gz"
        preview_kept: list[str] = []
        preview_rejected: list[str] = []
        pending: list[tuple[dict, list[str]]] = []

        with gzip.open(in_path, "rt", encoding="utf-8") as in_fh, gzip.open(
            out_path, "wt", encoding="utf-8"
        ) as out_fh:
            iterator = in_fh if args.max_poems is None else islice(in_fh, args.max_poems)
            for line in tqdm(iterator, desc=f"quality_{split}"):
                row = json.loads(line)
                stats[f"{split}_rows_total"] += 1
                pending.append((row, split_lines(row["text"])))
                if len(pending) >= args.stress_batch_size:
                    process_pending(
                        pending,
                        out_fh,
                        split,
                        stats,
                        reject_examples,
                        preview_kept,
                        preview_rejected,
                        args.min_score,
                        args.default_scheme,
                    )
                    pending.clear()

            process_pending(
                pending,
                out_fh,
                split,
                stats,
                reject_examples,
                preview_kept,
                preview_rejected,
                args.min_score,
                args.default_scheme,
            )

        (out_dir / f"{split}.kept.preview.txt").write_text("\n".join(preview_kept), encoding="utf-8")
        (out_dir / f"{split}.rejected.preview.txt").write_text(
            "\n".join(preview_rejected), encoding="utf-8"
        )

    summary = dict(stats)
    for split in ["train", "val", "test"]:
        total = stats.get(f"{split}_rows_total", 0)
        kept = stats.get(f"{split}_rows_kept", 0)
        summary[f"{split}_keep_rate"] = round(kept / total, 4) if total else 0.0
        summary[f"{split}_avg_score"] = round(stats.get(f"{split}_score_sum_x1000", 0) / max(total, 1) / 1000, 4)

    (out_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "reject_examples.json").write_text(
        json.dumps(reject_examples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
