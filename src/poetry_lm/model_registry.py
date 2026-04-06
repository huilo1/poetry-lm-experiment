from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    checkpoint: Path
    tokenizer: Path
    note: str


MODEL_SPECS = [
    ModelSpec(
        key="aabb_baseline",
        title="AABB CCDD baseline",
        checkpoint=Path("artifacts/checkpoints/host_5060_8line_20m/best.pt"),
        tokenizer=Path("artifacts/tokenizer_aabb8/poetry.model"),
        note="Текущий основной baseline под 8 строк и AABB CCDD.",
    ),
    ModelSpec(
        key="abab_branch",
        title="ABAB ABAB branch",
        checkpoint=Path("artifacts/checkpoints/host_5060_8line_abab_20m/best.pt"),
        tokenizer=Path("artifacts/tokenizer_abab8/poetry.model"),
        note="Сравнительная ветка с более частой схемой, но худшей рифмой через строку.",
    ),
    ModelSpec(
        key="staged_current",
        title="Stage1 -> Stage2 current",
        checkpoint=Path("artifacts/checkpoints/host_5060_aabb_qf2_stage2_from_fullpoem_20m/best.pt"),
        tokenizer=Path("artifacts/tokenizer_aabb8/poetry.model"),
        note="Текущий staged run: сначала полные стихи, потом строгий AABB CCDD.",
    ),
]


def model_specs() -> list[ModelSpec]:
    return MODEL_SPECS


def model_map() -> dict[str, ModelSpec]:
    return {spec.key: spec for spec in MODEL_SPECS}
