from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from poetry_lm.qwen_sft import (
    QwenSFTExample,
    build_full_text_from_generation,
    example_from_row,
    extract_generated_lines,
    normalize_line,
    normalize_lines,
    parse_labeled_poem,
    render_completion,
    render_prompt_prefix,
    render_training_text,
    split_poem_lines,
)


GIGACHAT_FORMAT_VERSION = "gigachat3_aabb_ccdd_v1"
DEFAULT_GIGACHAT_BASE_MODEL = "ai-sage/GigaChat3-10B-A1.8B-base"


def _model_cache_root() -> Path:
    return Path(os.environ.get("POETRY_LM_MODEL_CACHE", ".model_cache"))


def prepare_gigachat_local_model_dir(
    model_id: str = DEFAULT_GIGACHAT_BASE_MODEL,
    target_root: str | Path | None = None,
) -> str:
    cache_root = Path(target_root) if target_root is not None else _model_cache_root()
    slug = model_id.replace("/", "__")
    local_dir = cache_root / slug
    config_path = local_dir / "config.json"
    index_path = local_dir / "model.safetensors.index.json"
    marker_path = local_dir / ".gigachat_patched"
    has_weight_shards = any(local_dir.glob("model-*.safetensors"))

    if not config_path.exists() or not index_path.exists() or not has_weight_shards:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.md",
                "model-*.safetensors",
                "model.safetensors.index.json",
            ],
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    changed = False
    if isinstance(config.get("routed_scaling_factor"), int):
        config["routed_scaling_factor"] = float(config["routed_scaling_factor"])
        changed = True

    if changed or not marker_path.exists():
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        marker_path.write_text("patched\n", encoding="utf-8")

    return str(local_dir)


def format_dataset_row(row: dict) -> dict:
    example = example_from_row(row)
    prompt_text = render_prompt_prefix(example.prompt)
    completion_text = render_completion(example.completion_lines)
    return {
        "format_version": GIGACHAT_FORMAT_VERSION,
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


__all__ = [
    "QwenSFTExample",
    "GIGACHAT_FORMAT_VERSION",
    "build_full_text_from_generation",
    "example_from_row",
    "extract_generated_lines",
    "format_dataset_row",
    "normalize_line",
    "normalize_lines",
    "parse_labeled_poem",
    "prepare_gigachat_local_model_dir",
    "render_completion",
    "render_prompt_prefix",
    "render_training_text",
    "split_poem_lines",
]
