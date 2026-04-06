from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

URL = "https://huggingface.co/datasets/IlyaGusev/stihi_ru/resolve/main/stihi_ru.jsonl.zst"


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with output.open("wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=output.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                pbar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL)
    parser.add_argument("--output", default="data/raw/stihi_ru.jsonl.zst")
    args = parser.parse_args()
    download(args.url, Path(args.output))


if __name__ == "__main__":
    main()
