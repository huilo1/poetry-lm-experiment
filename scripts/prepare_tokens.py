from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from poetry_lm.tokenizer import Tokenizer, sample_to_training_text


def ids_dtype(vocab_size: int):
    return np.uint16 if vocab_size < 65535 else np.uint32


def encode_split(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> int:
    all_ids: list[int] = []
    with gzip.open(input_path, "rt", encoding="utf-8") as fh:
        for line in tqdm(fh, desc=f"encode_{input_path.stem}"):
            row = json.loads(line)
            formatted = sample_to_training_text(row)
            all_ids.extend(tokenizer.encode_formatted(formatted))
    arr = np.array(all_ids, dtype=ids_dtype(tokenizer.vocab_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(output_path)
    return len(arr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tokenizer-model", default="artifacts/tokenizer/poetry.model")
    args = parser.parse_args()

    if args.dataset_dir is not None:
        input_dir = Path(args.dataset_dir)
        output_dir = Path(args.dataset_dir)
    else:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir or args.input_dir)

    tokenizer = Tokenizer(args.tokenizer_model)
    counts = {}
    for split in ["train", "val", "test"]:
        counts[split] = encode_split(
            tokenizer, input_dir / f"{split}.jsonl.gz", output_dir / f"{split}.bin"
        )
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint16" if tokenizer.vocab_size < 65535 else "uint32",
        **counts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
