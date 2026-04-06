from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import jsonlines
import zstandard

from .rhyme import detect_quatrain_scheme
from .text import count_words, split_lines, stable_normalize_for_hash, text_quality_ok


@dataclass(slots=True)
class PoemRecord:
    source: str
    poem_id: str
    content_hash: str
    author: str | None
    title: str | None
    genre: str | None
    topic: str | None
    text: str
    line_count: int
    word_count: int
    rhyme_scheme: str | None
    rhyme_confidence: float


def iter_stihi_records(path: Path):
    with path.open("rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
            for row in jsonlines.Reader(text_stream):
                yield row


def poem_is_good(lines: list[str]) -> bool:
    if len(lines) < 4:
        return False
    avg_len = sum(len(line) for line in lines) / len(lines)
    if avg_len < 8 or avg_len > 80:
        return False
    if any(len(line) > 120 for line in lines):
        return False
    return True


def build_record(row: dict) -> PoemRecord | None:
    text = row.get("text") or ""
    if not text_quality_ok(text):
        return None
    lines = split_lines(text)
    if not poem_is_good(lines):
        return None
    word_count = count_words("\n".join(lines))
    if word_count < 20:
        return None
    scheme = detect_quatrain_scheme(lines[:4])
    if scheme.scheme is None:
        return None
    stable = stable_normalize_for_hash(lines)
    poem_hash = hashlib.sha1(stable.encode("utf-8")).hexdigest()
    return PoemRecord(
        source="stihi_ru",
        poem_id=row.get("id") or poem_hash,
        content_hash=poem_hash,
        author=(row.get("author") or None),
        title=(row.get("title") or None),
        genre=(row.get("genre") or None),
        topic=(row.get("topic") or None),
        text="\n".join(lines),
        line_count=len(lines),
        word_count=word_count,
        rhyme_scheme=scheme.scheme,
        rhyme_confidence=scheme.confidence,
    )


def write_jsonl_gz(path: Path, records: list[PoemRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def append_jsonl_gz_line(handle, record: PoemRecord) -> None:
    handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def deterministic_split_key(author: str | None, poem_id: str) -> float:
    base = f"{author or 'unknown'}::{poem_id}".encode("utf-8")
    digest = hashlib.sha1(base).digest()
    return int.from_bytes(digest[:8], byteorder="big") / 2**64
