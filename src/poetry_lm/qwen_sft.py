from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

QWEN_SCHEME = "AABB CCDD"
QWEN_FORMAT_VERSION = "qwen3_aabb_ccdd_v1"
LINE_TAG_RE = re.compile(r"\[L([1-8])\]\s*")


@dataclass(frozen=True)
class QwenSFTExample:
    prompt: str
    completion_lines: list[str]

    @property
    def all_lines(self) -> list[str]:
        return [self.prompt, *self.completion_lines]


def normalize_line(line: str) -> str:
    return " ".join(line.strip().split())


def normalize_lines(lines: Iterable[str]) -> list[str]:
    return [normalize_line(line) for line in lines if normalize_line(line)]


def split_poem_lines(text: str) -> list[str]:
    return normalize_lines(text.splitlines())


def example_from_row(row: dict) -> QwenSFTExample:
    lines = split_poem_lines(row["text"])
    if len(lines) != 8:
        raise ValueError(f"expected 8 lines, got {len(lines)}")
    return QwenSFTExample(prompt=lines[0], completion_lines=lines[1:])


def render_prompt_prefix(prompt: str) -> str:
    prompt = normalize_line(prompt)
    return "\n".join(
        [
            "[TASK] Продолжи русское стихотворение.",
            "[FORMAT] Ровно 8 строк.",
            f"[SCHEME] {QWEN_SCHEME}",
            f"[L1] {prompt}",
            "[GEN]",
            "[L2] ",
        ]
    )


def render_completion(completion_lines: list[str]) -> str:
    lines = normalize_lines(completion_lines)
    if len(lines) != 7:
        raise ValueError(f"expected 7 completion lines, got {len(lines)}")
    parts = [lines[0]]
    for index, line in enumerate(lines[1:], start=3):
        parts.append(f"\n[L{index}] {line}")
    return "".join(parts)


def render_training_text(example: QwenSFTExample) -> str:
    return render_prompt_prefix(example.prompt) + render_completion(example.completion_lines)


def format_dataset_row(row: dict) -> dict:
    example = example_from_row(row)
    prompt_text = render_prompt_prefix(example.prompt)
    completion_text = render_completion(example.completion_lines)
    return {
        "format_version": QWEN_FORMAT_VERSION,
        "prompt": example.prompt,
        "completion_lines": example.completion_lines,
        "prompt_text": prompt_text,
        "completion_text": completion_text,
        "train_text": prompt_text + completion_text,
        "rhyme_scheme": row.get("rhyme_scheme", "AABB_CCDD"),
        "source": row.get("source"),
        "poem_id": row.get("poem_id"),
        "author": row.get("author"),
        "title": row.get("title"),
        "quality_score": row.get("quality_score"),
    }


def parse_labeled_poem(text: str) -> list[str]:
    matches = list(LINE_TAG_RE.finditer(text))
    if not matches:
        return []
    lines_by_index: dict[int, str] = {}
    for idx, match in enumerate(matches):
        line_index = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        line = normalize_line(text[start:end])
        if line:
            lines_by_index[line_index] = line
    return [lines_by_index[index] for index in range(1, 9) if index in lines_by_index]


def build_full_text_from_generation(prompt: str, generated_continuation: str) -> str:
    return render_prompt_prefix(prompt) + generated_continuation


def extract_generated_lines(prompt: str, generated_continuation: str) -> list[str]:
    labeled = parse_labeled_poem(build_full_text_from_generation(prompt, generated_continuation))
    if len(labeled) >= 8:
        return labeled[:8]
    fallback = [normalize_line(prompt)]
    raw_lines = normalize_lines(generated_continuation.splitlines())
    fallback.extend(raw_lines)
    return fallback[:8]
