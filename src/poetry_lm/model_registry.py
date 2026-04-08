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
    adapter_dir: Path | None = None
    hf_base_model: str | None = None
    hf_load_in_4bit: bool = False
    hf_bf16: bool = False
    planner_checkpoint: Path | None = None
    refiner_checkpoint: Path | None = None
    refiner_tokenizer: Path | None = None

    @property
    def is_gigachat_lora(self) -> bool:
        return self.adapter_dir is not None and self.hf_base_model is not None

    @property
    def is_planner_guided(self) -> bool:
        return self.planner_checkpoint is not None

    @property
    def is_refiner_guided(self) -> bool:
        return self.refiner_checkpoint is not None

    def all_checkpoints(self) -> list[Path]:
        if self.is_gigachat_lora:
            return [
                self.checkpoint,
                self.adapter_dir / "adapter_config.json",
            ]
        paths = [self.checkpoint]
        if self.planner_checkpoint is not None:
            paths.insert(0, self.planner_checkpoint)
        if self.refiner_checkpoint is not None:
            paths.append(self.refiner_checkpoint)
        return paths

    def all_tokenizers(self) -> list[Path]:
        if self.is_gigachat_lora:
            return [
                self.tokenizer,
                self.adapter_dir / "tokenizer_config.json",
            ]
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
]


def model_specs() -> list[ModelSpec]:
    return MODEL_SPECS


def model_map() -> dict[str, ModelSpec]:
    return {spec.key: spec for spec in MODEL_SPECS}
