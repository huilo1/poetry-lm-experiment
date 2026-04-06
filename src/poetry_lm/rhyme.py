from __future__ import annotations

import re
from dataclasses import dataclass

from .stress import accent_last_word, accent_last_words, stress_position
from .text import VOWELS, last_word

FINAL_RE = re.compile(r"[^а-яё-]")

VOICELESS_MAP = str.maketrans(
    {
        "б": "п",
        "в": "ф",
        "г": "к",
        "д": "т",
        "ж": "ш",
        "з": "с",
    }
)


def normalize_word(word: str) -> str:
    word = word.lower().replace("ё", "е")
    word = FINAL_RE.sub("", word)
    word = word.strip("-")
    return word.translate(VOICELESS_MAP)


def rhyme_tail_from_word(word: str) -> str:
    word = normalize_word(word)
    if not word:
        return ""
    vowel_positions = [idx for idx, ch in enumerate(word) if ch in VOWELS]
    if not vowel_positions:
        return word[-3:]
    pos = vowel_positions[-1]
    tail = word[pos:]
    if len(tail) < 2 and len(vowel_positions) > 1:
        tail = word[vowel_positions[-2] :]
    return tail


def stressed_rhyme_tail_from_word(word: str) -> str:
    normalized = normalize_word(word.replace("'", ""))
    if not normalized:
        return ""
    stress_idx = stress_position(word)
    if stress_idx is None:
        return rhyme_tail_from_word(word)

    raw = word.lower().replace("'", "")
    compact = []
    raw_to_compact: dict[int, int] = {}
    for idx, ch in enumerate(raw):
        if ch.isalpha() or ch == "-":
            raw_to_compact[idx] = len(compact)
            compact.append(ch)
    if stress_idx not in raw_to_compact:
        return rhyme_tail_from_word(word)

    compact_idx = raw_to_compact[stress_idx]
    normalized_idx = 0
    for idx, ch in enumerate(compact):
        if idx == compact_idx:
            normalized_idx = idx
            break
    tail = normalize_word("".join(compact[normalized_idx:]))
    return tail or rhyme_tail_from_word(word)


def rhyme_tail(line: str, use_stress: bool = False) -> str:
    if use_stress:
        tail = stressed_rhyme_tail_from_word(accent_last_word(line))
        if tail:
            return tail
    return rhyme_tail_from_word(last_word(line))


def lines_rhyme_tails(lines: list[str], use_stress: bool = False) -> list[str]:
    if not use_stress:
        return [rhyme_tail_from_word(last_word(line)) for line in lines]

    accented = accent_last_words(lines)
    return lines_rhyme_tails_from_accented(lines, accented)


def lines_rhyme_tails_from_accented(lines: list[str], accented: list[str]) -> list[str]:
    tails = []
    for line, accented_word in zip(lines, accented):
        tail = stressed_rhyme_tail_from_word(accented_word)
        tails.append(tail or rhyme_tail_from_word(last_word(line)))
    return tails


def tails_rhyme(l_tail: str, r_tail: str) -> bool:
    if not l_tail or not r_tail:
        return False
    if l_tail == r_tail:
        return True
    if len(l_tail) >= 2 and len(r_tail) >= 2 and l_tail[-2:] == r_tail[-2:]:
        return True
    if len(l_tail) >= 3 and len(r_tail) >= 3 and l_tail[-3:] == r_tail[-3:]:
        return True
    return False


def rhymes(left: str, right: str, use_stress: bool = False) -> bool:
    l_tail = rhyme_tail(left, use_stress=use_stress)
    r_tail = rhyme_tail(right, use_stress=use_stress)
    return tails_rhyme(l_tail, r_tail)


@dataclass(slots=True)
class QuatrainRhyme:
    scheme: str | None
    confidence: float


def detect_quatrain_scheme_from_tails(tails: list[str]) -> QuatrainRhyme:
    if len(tails) < 4:
        return QuatrainRhyme(None, 0.0)
    a1 = tails_rhyme(tails[0], tails[1])
    a2 = tails_rhyme(tails[2], tails[3])
    b1 = tails_rhyme(tails[0], tails[2])
    b2 = tails_rhyme(tails[1], tails[3])
    if a1 and a2:
        return QuatrainRhyme("AABB", 0.9 if not (b1 or b2) else 0.7)
    if b1 and b2:
        return QuatrainRhyme("ABAB", 0.9 if not (a1 or a2) else 0.7)
    if tails_rhyme(tails[0], tails[3]) and tails_rhyme(tails[1], tails[2]):
        return QuatrainRhyme("ABBA", 0.8)
    return QuatrainRhyme(None, 0.0)


def detect_quatrain_scheme(lines: list[str], use_stress: bool = False) -> QuatrainRhyme:
    if len(lines) < 4:
        return QuatrainRhyme(None, 0.0)
    tails = lines_rhyme_tails(lines[:4], use_stress=use_stress)
    return detect_quatrain_scheme_from_tails(tails)


def detect_eight_line_aabb_aabb_from_tails(tails: list[str]) -> QuatrainRhyme:
    if len(tails) < 8:
        return QuatrainRhyme(None, 0.0)
    first = detect_quatrain_scheme_from_tails(tails[:4])
    second = detect_quatrain_scheme_from_tails(tails[4:8])
    if first.scheme != "AABB" or second.scheme != "AABB":
        return QuatrainRhyme(None, 0.0)
    return QuatrainRhyme("AABB_CCDD", min(first.confidence, second.confidence))


def detect_eight_line_aabb_aabb(lines: list[str], use_stress: bool = False) -> QuatrainRhyme:
    if len(lines) < 8:
        return QuatrainRhyme(None, 0.0)
    tails = lines_rhyme_tails(lines[:8], use_stress=use_stress)
    return detect_eight_line_aabb_aabb_from_tails(tails)


def detect_eight_line_abab_abab_from_tails(tails: list[str]) -> QuatrainRhyme:
    if len(tails) < 8:
        return QuatrainRhyme(None, 0.0)
    first = detect_quatrain_scheme_from_tails(tails[:4])
    second = detect_quatrain_scheme_from_tails(tails[4:8])
    if first.scheme != "ABAB" or second.scheme != "ABAB":
        return QuatrainRhyme(None, 0.0)
    return QuatrainRhyme("ABAB_ABAB", min(first.confidence, second.confidence))


def detect_eight_line_abab_abab(lines: list[str], use_stress: bool = False) -> QuatrainRhyme:
    if len(lines) < 8:
        return QuatrainRhyme(None, 0.0)
    tails = lines_rhyme_tails(lines[:8], use_stress=use_stress)
    return detect_eight_line_abab_abab_from_tails(tails)
