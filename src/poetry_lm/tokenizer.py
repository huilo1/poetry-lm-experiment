from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

STRUCTURED_MODE = "structured_8line_aabb_ccdd"
STRUCTURED_MODE_ABAB = "structured_8line_abab_abab"
STRUCTURED_SCHEME = "AABB_CCDD"
STRUCTURED_SCHEME_ABAB = "ABAB_ABAB"
STRUCTURED_TOKENS = [
    "<PROMPT>",
    "<GEN>",
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
]


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

    def encode_poem(self, text: str) -> list[int]:
        return self.sp.encode(poem_to_training_text(text), out_type=int, add_bos=True, add_eos=True)

    def encode_formatted(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int, add_bos=True, add_eos=True)

    def format_prompt(self, line: str, mode: str = "legacy") -> str:
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
        return line.strip() + " <NL>"

    def encode_prompt(self, line: str, mode: str = "legacy") -> list[int]:
        if mode in {STRUCTURED_MODE, STRUCTURED_MODE_ABAB}:
            return self.sp.encode(self.format_prompt(line, mode=mode), out_type=int, add_bos=True, add_eos=False)
        return self.sp.encode(line.strip() + " <NL>", out_type=int, add_bos=True, add_eos=False)

    def decode(self, ids: list[int], mode: str = "legacy") -> str:
        text = self.sp.decode(ids)
        if mode in {STRUCTURED_MODE, STRUCTURED_MODE_ABAB}:
            for token in ["<PROMPT>", "<GEN>", "<LEN_8>", "<SCHEME_AABB_CCDD>", "<SCHEME_ABAB_ABAB>"]:
                text = text.replace(token, " ")
            text = text.replace("<L1>", " ")
            for idx in range(2, 9):
                text = text.replace(f"<L{idx}>", "\n")
        text = text.replace("<NL>", "\n")
        lines = [" ".join(line.split()) for line in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()
