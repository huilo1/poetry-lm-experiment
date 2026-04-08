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
    planner_checkpoint: Path | None = None
    refiner_checkpoint: Path | None = None
    refiner_tokenizer: Path | None = None

    @property
    def is_planner_guided(self) -> bool:
        return self.planner_checkpoint is not None

    @property
    def is_refiner_guided(self) -> bool:
        return self.refiner_checkpoint is not None

    def all_checkpoints(self) -> list[Path]:
        paths = [self.checkpoint]
        if self.planner_checkpoint is not None:
            paths.insert(0, self.planner_checkpoint)
        if self.refiner_checkpoint is not None:
            paths.append(self.refiner_checkpoint)
        return paths

    def all_tokenizers(self) -> list[Path]:
        paths = [self.tokenizer]
        if self.refiner_tokenizer is not None:
            paths.append(self.refiner_tokenizer)
        return paths


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
        key="aabb_planner_guided",
        title="AABB planner-guided",
        checkpoint=Path("artifacts/checkpoints/host_5060_aabb_with_plan_20m/best.pt"),
        planner_checkpoint=Path("artifacts/checkpoints/host_5060_aabb_end_planner_12m/best.pt"),
        tokenizer=Path("artifacts/tokenizer_aabb_plan/poetry.model"),
        note="Новая ветка: planner сначала предсказывает окончания строк 2/4/6/8, затем generator пишет стих под этот план.",
    ),
    ModelSpec(
        key="aabb_refiner",
        title="AABB refiner-guided",
        checkpoint=Path("artifacts/checkpoints/host_5060_8line_20m/best.pt"),
        tokenizer=Path("artifacts/tokenizer_aabb8/poetry.model"),
        refiner_checkpoint=Path("artifacts/checkpoints/vast_refiner_aabb_20m/best.pt"),
        refiner_tokenizer=Path("artifacts/tokenizer_refiner/poetry.model"),
        note="Diffusion-style refiner: baseline сначала пишет черновик, затем bidirectional denoiser пытается отредактировать текст.",
    ),
]


def model_specs() -> list[ModelSpec]:
    return MODEL_SPECS


def model_map() -> dict[str, ModelSpec]:
    return {spec.key: spec for spec in MODEL_SPECS}
