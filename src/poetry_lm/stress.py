from __future__ import annotations

import re
from functools import lru_cache

import torch

try:
    from ruaccent import load_accentor
except ImportError:  # pragma: no cover - optional runtime dependency in old checkpoints
    load_accentor = None

from .text import last_word

ACCENT_MARK = "'"
ACCENTED_WORD_RE = re.compile(r"[А-Яа-яЁё']+")
VOWELS = "аеёиоуыэюя"


@lru_cache(maxsize=1)
def get_accentor():
    if load_accentor is None:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_accentor(device=device)


def stress_available() -> bool:
    return get_accentor() is not None


@lru_cache(maxsize=200_000)
def accent_word(word: str) -> str | None:
    accentor = get_accentor()
    if accentor is None:
        return None
    normalized = word.strip()
    if not normalized:
        return None
    if normalized[-1].isalpha():
        normalized = normalized + "."
    try:
        return accentor.put_accent(normalized)
    except Exception:
        return None


def accent_words(words: list[str]) -> list[str]:
    accentor = get_accentor()
    if accentor is None:
        return ["" for _ in words]

    pending_inputs: list[str] = []
    pending_indices: list[int] = []
    results = ["" for _ in words]
    for idx, word in enumerate(words):
        normalized = word.strip()
        if not normalized:
            continue
        if normalized[-1].isalpha():
            normalized = normalized + "."
        pending_inputs.append(normalized)
        pending_indices.append(idx)

    if not pending_inputs:
        return results

    try:
        accented_batch = accentor.put_accent(pending_inputs)
        if isinstance(accented_batch, str):
            accented_batch = [accented_batch]
    except Exception:
        accented_batch = []

    if len(accented_batch) == len(pending_indices):
        for idx, accented in zip(pending_indices, accented_batch):
            results[idx] = last_accented_word(accented or "")
        return results

    for idx, word in enumerate(words):
        results[idx] = last_accented_word(accent_word(word) or "")
    return results


def last_accented_word(text: str) -> str:
    words = ACCENTED_WORD_RE.findall(text.lower().replace("—", " ").replace("–", " "))
    return words[-1] if words else ""


def accent_last_word(line: str) -> str:
    word = last_word(line)
    if not word:
        return ""
    accented = accent_word(word)
    return last_accented_word(accented or "")


def accent_last_words(lines: list[str]) -> list[str]:
    return accent_words([last_word(line) for line in lines])


def stress_position(word: str) -> int | None:
    clean = word.lower()
    if "ё" in clean:
        return clean.index("ё")
    for idx, ch in enumerate(clean):
        if ch == ACCENT_MARK and idx > 0 and clean[idx - 1] in VOWELS:
            return idx - 1
    vowel_positions = [idx for idx, ch in enumerate(clean) if ch in VOWELS]
    if len(vowel_positions) == 1:
        return vowel_positions[0]
    return None
