from __future__ import annotations

import re
from typing import Iterable

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z-]+")
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTIBLANK_RE = re.compile(r"\n{3,}")
NON_TEXT_RE = re.compile(r"(https?://|<[^>]+>|&\w+;)")
VOWELS = "аеёиоуыэюя"


def normalize_line(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip(" \t")


def normalize_poem(text: str) -> str:
    lines = [normalize_line(line) for line in text.split("\n")]
    lines = [line for line in lines if line]
    text = "\n".join(lines)
    return MULTIBLANK_RE.sub("\n\n", text).strip()


def is_mostly_cyrillic(text: str, min_ratio: float = 0.6) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    cyr = sum(1 for ch in letters if CYRILLIC_RE.match(ch))
    return cyr / len(letters) >= min_ratio


def text_quality_ok(text: str) -> bool:
    if NON_TEXT_RE.search(text):
        return False
    if not is_mostly_cyrillic(text):
        return False
    return True


def split_lines(text: str) -> list[str]:
    return [line.strip() for line in normalize_poem(text).splitlines() if line.strip()]


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def last_word(line: str) -> str:
    words = WORD_RE.findall(line.lower().replace("—", " ").replace("–", " "))
    return words[-1] if words else ""


def stable_normalize_for_hash(lines: Iterable[str]) -> str:
    parts = []
    for line in lines:
        line = re.sub(r"[^\w\s]", "", line.lower())
        line = MULTISPACE_RE.sub(" ", line).strip()
        if line:
            parts.append(line)
    return "\n".join(parts)
