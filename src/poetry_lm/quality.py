from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .rhyme import lines_rhyme_tails, normalize_word, tails_rhyme
from .text import VOWELS, WORD_RE, is_mostly_cyrillic, last_word

BAD_CHAR_RE = re.compile(r"[�⁇]")
CONTINUATION_START_RE = re.compile(r"^[\-\—\–,;:\)\]\}…\.]")
FRAGMENT_END_RE = re.compile(r"[\-—–,:;\(\[\{]$")


@dataclass(slots=True)
class WindowQuality:
    keep: bool
    score: float
    hard_reasons: list[str]
    soft_reasons: list[str]
    features: dict[str, float | int | str]


def line_syllables(line: str) -> int:
    return sum(1 for ch in line.lower() if ch in VOWELS)


def line_words(line: str) -> list[str]:
    return WORD_RE.findall(line.lower())


def normalized_line(line: str) -> str:
    return " ".join(line_words(line))


def lexical_diversity(lines: list[str]) -> float:
    words = [word for line in lines for word in line_words(line)]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def line_repeat_ratio(lines: list[str]) -> float:
    normalized = [normalized_line(line) for line in lines if normalized_line(line)]
    if not normalized:
        return 1.0
    return len(set(normalized)) / len(normalized)


def content_word_repeat_ratio(lines: list[str]) -> float:
    words = [word for line in lines for word in line_words(line) if len(word) >= 4]
    if not words:
        return 0.0
    counts = Counter(words)
    return counts.most_common(1)[0][1] / len(words)


def scheme_pair_indices(scheme: str) -> list[tuple[int, int]]:
    if scheme == "ABAB_ABAB":
        return [(0, 2), (1, 3), (4, 6), (5, 7)]
    return [(0, 1), (2, 3), (4, 5), (6, 7)]


def score_window(lines: list[str], tails: list[str] | None = None, scheme: str = "AABB_CCDD") -> WindowQuality:
    hard_reasons: list[str] = []
    soft_reasons: list[str] = []
    features: dict[str, float | int | str] = {}

    if len(lines) != 8:
        hard_reasons.append("wrong_line_count")
        return WindowQuality(False, 0.0, hard_reasons, soft_reasons, features)

    stripped = [line.strip() for line in lines]
    if any(not line for line in stripped):
        hard_reasons.append("empty_line")

    line_lengths = [len(line) for line in stripped]
    word_counts = [len(line_words(line)) for line in stripped]
    syllables = [line_syllables(line) for line in stripped]
    last_words = [normalize_word(last_word(line)) for line in stripped]
    tails = tails or lines_rhyme_tails(stripped, use_stress=True)

    features["avg_line_len"] = round(sum(line_lengths) / len(line_lengths), 2)
    features["max_line_len"] = max(line_lengths)
    features["min_line_len"] = min(line_lengths)
    features["avg_word_count"] = round(sum(word_counts) / len(word_counts), 2)
    features["lexical_diversity"] = round(lexical_diversity(stripped), 4)
    features["line_repeat_ratio"] = round(line_repeat_ratio(stripped), 4)
    features["content_word_repeat_ratio"] = round(content_word_repeat_ratio(stripped), 4)

    if any(BAD_CHAR_RE.search(line) for line in stripped):
        hard_reasons.append("bad_chars")
    if any(not is_mostly_cyrillic(line, min_ratio=0.8) for line in stripped):
        hard_reasons.append("non_cyrillic_line")
    if min(line_lengths) < 8 or max(line_lengths) > 96:
        hard_reasons.append("line_length_extreme")
    if min(word_counts) < 2:
        hard_reasons.append("too_few_words")
    if min(syllables) < 3 or max(syllables) > 20:
        hard_reasons.append("syllable_extreme")
    if CONTINUATION_START_RE.search(stripped[0]):
        hard_reasons.append("continuation_start")
    if FRAGMENT_END_RE.search(stripped[-1]):
        hard_reasons.append("fragment_end")

    unique_lines = {normalized_line(line) for line in stripped}
    if len(unique_lines) < len(stripped):
        hard_reasons.append("duplicate_line")

    pair_indices = scheme_pair_indices(scheme)
    pair_syllable_diffs: list[int] = []
    cross_collisions = 0
    exact_last_word_collisions = 0

    for left, right in pair_indices:
        if not tails_rhyme(tails[left], tails[right]):
            hard_reasons.append(f"missing_pair_rhyme_{left+1}_{right+1}")
        if last_words[left] and last_words[left] == last_words[right]:
            exact_last_word_collisions += 1
        pair_syllable_diffs.append(abs(syllables[left] - syllables[right]))

    features["max_pair_syllable_diff"] = max(pair_syllable_diffs)
    features["avg_pair_syllable_diff"] = round(sum(pair_syllable_diffs) / len(pair_syllable_diffs), 2)

    allowed_pairs = {tuple(sorted(pair)) for pair in pair_indices}
    for left in range(8):
        for right in range(left + 1, 8):
            if (left, right) in allowed_pairs:
                continue
            if tails_rhyme(tails[left], tails[right]):
                cross_collisions += 1

    features["cross_rhyme_collisions"] = cross_collisions
    features["exact_last_word_collisions"] = exact_last_word_collisions

    if max(pair_syllable_diffs) > 7:
        hard_reasons.append("pair_meter_mismatch")
    if exact_last_word_collisions > 0:
        hard_reasons.append("same_last_word_in_pair")

    score = 1.0

    if max(pair_syllable_diffs) <= 2:
        score += 0.25
    elif max(pair_syllable_diffs) <= 4:
        score += 0.12
    else:
        soft_reasons.append("loose_pair_meter")
        score -= 0.08

    diversity = lexical_diversity(stripped)
    if diversity >= 0.72:
        score += 0.18
    elif diversity >= 0.58:
        score += 0.1
    else:
        soft_reasons.append("low_lexical_diversity")
        score -= 0.08

    repeat_ratio = content_word_repeat_ratio(stripped)
    if repeat_ratio <= 0.08:
        score += 0.14
    elif repeat_ratio <= 0.12:
        score += 0.06
    else:
        soft_reasons.append("high_word_repetition")
        score -= min(0.16, (repeat_ratio - 0.12) * 2)

    if line_repeat_ratio(stripped) == 1.0:
        score += 0.12
    else:
        soft_reasons.append("line_repetition")
        score -= 0.12

    avg_len = sum(line_lengths) / len(line_lengths)
    if 18 <= avg_len <= 48:
        score += 0.1
    else:
        soft_reasons.append("line_length_skew")
        score -= 0.06

    if cross_collisions == 0:
        score += 0.14
    elif cross_collisions == 1:
        score += 0.04
    else:
        soft_reasons.append("cross_rhyme_collision")
        score -= min(0.18, 0.05 * cross_collisions)

    score = max(0.0, min(1.5, score))
    keep = not hard_reasons
    return WindowQuality(keep, round(score, 4), hard_reasons, soft_reasons, features)
