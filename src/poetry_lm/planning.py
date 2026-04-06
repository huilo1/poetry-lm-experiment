from __future__ import annotations

from poetry_lm.text import last_word, split_lines
from poetry_lm.tokenizer import PLAN_TAGS, normalize_plan_endings


def ending_words_from_lines(lines: list[str]) -> dict[str, str]:
    if len(lines) != 8:
        raise ValueError("expected exactly 8 lines")
    return normalize_plan_endings(
        {
            "<E2>": last_word(lines[1]),
            "<E4>": last_word(lines[3]),
            "<E6>": last_word(lines[5]),
            "<E8>": last_word(lines[7]),
        }
    )


def ending_words_from_text(text: str) -> dict[str, str]:
    return ending_words_from_lines(split_lines(text))


def plan_is_complete(plan_endings: dict[str, str] | None) -> bool:
    if not plan_endings:
        return False
    normalized = normalize_plan_endings(plan_endings)
    return all(normalized[tag] for tag in PLAN_TAGS)
