from __future__ import annotations

from pathlib import Path
import re

import sentencepiece as spm

STRUCTURED_MODE = "structured_8line_aabb_ccdd"
STRUCTURED_MODE_ABAB = "structured_8line_abab_abab"
STRUCTURED_MODE_AABB_PLAN = "structured_8line_aabb_ccdd_plan"
PLANNER_MODE_AABB = "planner_8line_aabb_ccdd"
STRUCTURED_SCHEME = "AABB_CCDD"
STRUCTURED_SCHEME_ABAB = "ABAB_ABAB"
STRUCTURED_TOKENS = [
    "<PROMPT>",
    "<GEN>",
    "<PLAN>",
    "<MASK>",
    "<LEN_8>",
    "<SCHEME_AABB_CCDD>",
    "<SCHEME_ABAB_ABAB>",
    "<L1>",
    "<L2>",
    "<L3>",
    "<L4>",
    "<L5>",
    "<L6>",
    "<L7>",
    "<L8>",
    "<E2>",
    "<E4>",
    "<E6>",
    "<E8>",
]
PLAN_TAGS = ("<E2>", "<E4>", "<E6>", "<E8>")
PLAN_SECTION_RE = re.compile(r"<PLAN>.*?<GEN>", re.DOTALL)


def poem_to_training_text(poem_text: str) -> str:
    return poem_text.replace("\n", " <NL> ")


def scheme_token_for_mode(mode: str) -> str:
    if mode == STRUCTURED_MODE_ABAB:
        return "<SCHEME_ABAB_ABAB>"
    return "<SCHEME_AABB_CCDD>"


def mode_for_scheme(scheme: str) -> str:
    if scheme == STRUCTURED_SCHEME_ABAB:
        return STRUCTURED_MODE_ABAB
    return STRUCTURED_MODE


def normalize_plan_word(word: str) -> str:
    return " ".join(word.strip().lower().split())


def normalize_plan_endings(plan_endings: dict[str, str] | None) -> dict[str, str]:
    normalized = {tag: "" for tag in PLAN_TAGS}
    if not plan_endings:
        return normalized
    for tag in PLAN_TAGS:
        normalized[tag] = normalize_plan_word(plan_endings.get(tag, ""))
    return normalized


def plan_tokens(plan_endings: dict[str, str] | None) -> list[str]:
    parts = ["<PLAN>"]
    normalized = normalize_plan_endings(plan_endings)
    for tag in PLAN_TAGS:
        parts.append(tag)
        if normalized[tag]:
            parts.append(normalized[tag])
    return parts


def structured_window_to_training_text(lines: list[str], scheme: str = STRUCTURED_SCHEME) -> str:
    if len(lines) != 8:
        raise ValueError("structured window must contain exactly 8 lines")
    mode = mode_for_scheme(scheme)
    parts = ["<PROMPT>", "<LEN_8>", scheme_token_for_mode(mode), "<L1>", lines[0], "<NL>", "<GEN>"]
    for idx, line in enumerate(lines[1:], start=2):
        parts.extend([f"<L{idx}>", line])
        if idx < 8:
            parts.append("<NL>")
    return " ".join(parts)


def planner_window_to_training_text(lines: list[str], plan_endings: dict[str, str]) -> str:
    if len(lines) != 8:
        raise ValueError("planner window must contain exactly 8 lines")
    parts = ["<PROMPT>", "<LEN_8>", "<SCHEME_AABB_CCDD>", "<L1>", lines[0], "<NL>"]
    parts.extend(plan_tokens(plan_endings))
    return " ".join(parts)


def structured_window_to_training_text_with_plan(lines: list[str], plan_endings: dict[str, str]) -> str:
    if len(lines) != 8:
        raise ValueError("structured window must contain exactly 8 lines")
    parts = ["<PROMPT>", "<LEN_8>", "<SCHEME_AABB_CCDD>", "<L1>", lines[0], "<NL>"]
    parts.extend(plan_tokens(plan_endings))
    parts.extend(["<GEN>", "<L2>", lines[1]])
    for idx, line in enumerate(lines[2:], start=3):
        parts.extend(["<NL>", f"<L{idx}>", line])
    return " ".join(parts)


def sample_to_training_text(row: dict) -> str:
    return row.get("formatted_text") or poem_to_training_text(row["text"])


def control_tokens() -> list[str]:
    return ["<NL>", *STRUCTURED_TOKENS]


def train_sentencepiece(
    input_path: Path, model_prefix: Path, vocab_size: int, user_defined_symbols: list[str] | None = None
) -> None:
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    symbols = user_defined_symbols or control_tokens()
    spm.SentencePieceTrainer.Train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=",".join(symbols),
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
    )


class Tokenizer:
    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    def nl_id(self) -> int:
        return self.sp.piece_to_id("<NL>")

    def mask_id(self) -> int:
        return self.sp.piece_to_id("<MASK>")

    def piece_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)

    def piece(self, idx: int) -> str:
        return self.sp.id_to_piece(idx)

    def encode_poem(self, text: str) -> list[int]:
        return self.sp.encode(poem_to_training_text(text), out_type=int, add_bos=True, add_eos=True)

    def encode_formatted(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int, add_bos=True, add_eos=True)

    def format_prompt(
        self, line: str, mode: str = "legacy", plan_endings: dict[str, str] | None = None
    ) -> str:
        if mode in {STRUCTURED_MODE, STRUCTURED_MODE_ABAB}:
            return " ".join(
                [
                    "<PROMPT>",
                    "<LEN_8>",
                    scheme_token_for_mode(mode),
                    "<L1>",
                    line.strip(),
                    "<NL>",
                    "<GEN>",
                    "<L2>",
                ]
            )
        if mode == STRUCTURED_MODE_AABB_PLAN:
            return " ".join(
                [
                    "<PROMPT>",
                    "<LEN_8>",
                    "<SCHEME_AABB_CCDD>",
                    "<L1>",
                    line.strip(),
                    "<NL>",
                    *plan_tokens(plan_endings),
                    "<GEN>",
                    "<L2>",
                ]
            )
        if mode == PLANNER_MODE_AABB:
            return " ".join(
                [
                    "<PROMPT>",
                    "<LEN_8>",
                    "<SCHEME_AABB_CCDD>",
                    "<L1>",
                    line.strip(),
                    "<NL>",
                    "<PLAN>",
                ]
            )
        return line.strip() + " <NL>"

    def encode_prompt(
        self, line: str, mode: str = "legacy", plan_endings: dict[str, str] | None = None
    ) -> list[int]:
        if mode in {STRUCTURED_MODE, STRUCTURED_MODE_ABAB, STRUCTURED_MODE_AABB_PLAN, PLANNER_MODE_AABB}:
            return self.sp.encode(
                self.format_prompt(line, mode=mode, plan_endings=plan_endings),
                out_type=int,
                add_bos=True,
                add_eos=False,
            )
        return self.sp.encode(line.strip() + " <NL>", out_type=int, add_bos=True, add_eos=False)

    def decode_raw(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    def decode_plan(self, ids: list[int]) -> dict[str, str]:
        text = self.decode_raw(ids)
        plan = {tag: "" for tag in PLAN_TAGS}
        for idx, tag in enumerate(PLAN_TAGS):
            start = text.find(tag)
            if start < 0:
                continue
            start += len(tag)
            next_positions = [text.find(next_tag, start) for next_tag in PLAN_TAGS[idx + 1 :]]
            gen_pos = text.find("<GEN>", start)
            next_positions = [pos for pos in next_positions if pos >= 0]
            if gen_pos >= 0:
                next_positions.append(gen_pos)
            end = min(next_positions) if next_positions else len(text)
            plan[tag] = normalize_plan_word(text[start:end])
        return plan

    def decode(self, ids: list[int], mode: str = "legacy") -> str:
        text = self.sp.decode(ids)
        if mode == STRUCTURED_MODE_AABB_PLAN:
            text = PLAN_SECTION_RE.sub("<GEN>", text)
        if mode in {STRUCTURED_MODE, STRUCTURED_MODE_ABAB, STRUCTURED_MODE_AABB_PLAN}:
            for token in ["<PROMPT>", "<GEN>", "<PLAN>", "<LEN_8>", "<SCHEME_AABB_CCDD>", "<SCHEME_ABAB_ABAB>"]:
                text = text.replace(token, " ")
            for token in PLAN_TAGS:
                text = text.replace(token, " ")
            text = text.replace("<L1>", " ")
            for idx in range(2, 9):
                text = text.replace(f"<L{idx}>", "\n")
        elif mode == PLANNER_MODE_AABB:
            for token in ["<PROMPT>", "<LEN_8>", "<SCHEME_AABB_CCDD>", "<L1>", "<PLAN>", "<NL>"]:
                text = text.replace(token, " ")
            return " ".join(text.split()).strip()
        text = text.replace("<NL>", "\n")
        lines = [" ".join(line.split()) for line in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()
