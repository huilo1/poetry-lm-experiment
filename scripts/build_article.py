from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from jinja2 import Template


ROOT = Path(__file__).resolve().parent.parent
ARTICLE_DIR = ROOT / "article"
OUTPUT_HTML = ARTICLE_DIR / "poetry_lm_experiment_report.html"


def load_json_with_preamble(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in {path}")
    return json.loads(text[start:])


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def format_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def svg_line_chart(
    series: list[dict],
    title: str,
    width: int = 760,
    height: int = 340,
    y_label: str = "val_loss",
) -> str:
    margin = {"top": 30, "right": 20, "bottom": 42, "left": 56}
    inner_w = width - margin["left"] - margin["right"]
    inner_h = height - margin["top"] - margin["bottom"]

    xs = [point[0] for item in series for point in item["points"]]
    ys = [point[1] for item in series for point in item["points"]]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1e-6
    pad_y = (max_y - min_y) * 0.08
    min_y -= pad_y
    max_y += pad_y

    def sx(x: float) -> float:
        return margin["left"] + (x - min_x) / (max_x - min_x) * inner_w

    def sy(y: float) -> float:
        return margin["top"] + inner_h - (y - min_y) / (max_y - min_y) * inner_h

    def path(points: list[tuple[float, float]]) -> str:
        return " ".join(
            ("M" if idx == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}" for idx, (x, y) in enumerate(points)
        )

    y_ticks = 5
    x_ticks = 6
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">',
        '<rect width="100%" height="100%" fill="white" rx="16" />',
        f'<text x="{margin["left"]}" y="20" font-size="16" font-weight="700" fill="#1f2937">{title}</text>',
    ]

    for i in range(y_ticks + 1):
        yv = min_y + (max_y - min_y) * i / y_ticks
        yy = sy(yv)
        parts.append(f'<line x1="{margin["left"]}" y1="{yy:.1f}" x2="{width - margin["right"]}" y2="{yy:.1f}" stroke="#e5e7eb" />')
        parts.append(f'<text x="{margin["left"] - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="11" fill="#6b7280">{yv:.2f}</text>')

    for i in range(x_ticks + 1):
        xv = min_x + (max_x - min_x) * i / x_ticks
        xx = sx(xv)
        parts.append(f'<line x1="{xx:.1f}" y1="{margin["top"]}" x2="{xx:.1f}" y2="{margin["top"] + inner_h}" stroke="#f3f4f6" />')
        parts.append(f'<text x="{xx:.1f}" y="{height - 12}" text-anchor="middle" font-size="11" fill="#6b7280">{int(xv)}</text>')

    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + inner_h}" x2="{width - margin["right"]}" y2="{margin["top"] + inner_h}" stroke="#9ca3af" />')
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + inner_h}" stroke="#9ca3af" />')
    parts.append(f'<text x="{width/2:.1f}" y="{height - 2}" text-anchor="middle" font-size="12" fill="#6b7280">Итерация</text>')
    parts.append(f'<text x="16" y="{height/2:.1f}" transform="rotate(-90 16 {height/2:.1f})" text-anchor="middle" font-size="12" fill="#6b7280">{y_label}</text>')

    legend_y = margin["top"] + 6
    legend_x = width - margin["right"] - 180
    for idx, item in enumerate(series):
        ly = legend_y + idx * 18
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 18}" y2="{ly}" stroke="{item["color"]}" stroke-width="3" />')
        parts.append(f'<text x="{legend_x + 24}" y="{ly + 4}" font-size="11" fill="#374151">{item["label"]}</text>')
        parts.append(
            f'<path d="{path(item["points"])}" fill="none" stroke="{item["color"]}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
        )
    parts.append("</svg>")
    return "".join(parts)


def svg_grouped_bar_chart(
    categories: list[str],
    series: list[dict],
    title: str,
    width: int = 760,
    height: int = 360,
) -> str:
    margin = {"top": 34, "right": 20, "bottom": 76, "left": 56}
    inner_w = width - margin["left"] - margin["right"]
    inner_h = height - margin["top"] - margin["bottom"]
    max_y = max(max(item["values"]) for item in series)
    max_y = max(max_y * 1.12, 0.1)

    group_w = inner_w / max(len(categories), 1)
    bar_w = group_w / (len(series) + 1)

    def sy(y: float) -> float:
        return margin["top"] + inner_h - y / max_y * inner_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">',
        '<rect width="100%" height="100%" fill="white" rx="16" />',
        f'<text x="{margin["left"]}" y="20" font-size="16" font-weight="700" fill="#1f2937">{title}</text>',
    ]

    for i in range(6):
        yv = max_y * i / 5
        yy = sy(yv)
        parts.append(f'<line x1="{margin["left"]}" y1="{yy:.1f}" x2="{width - margin["right"]}" y2="{yy:.1f}" stroke="#e5e7eb" />')
        parts.append(f'<text x="{margin["left"] - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="11" fill="#6b7280">{yv:.2f}</text>')

    for ci, cat in enumerate(categories):
        base_x = margin["left"] + ci * group_w
        for si, item in enumerate(series):
            value = item["values"][ci]
            x = base_x + si * bar_w + bar_w * 0.2
            y = sy(value)
            h = margin["top"] + inner_h - y
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.6:.1f}" height="{h:.1f}" fill="{item["color"]}" rx="4" />')
        parts.append(
            f'<text x="{base_x + group_w/2:.1f}" y="{height - 34}" text-anchor="middle" font-size="11" fill="#374151" transform="rotate(-20 {base_x + group_w/2:.1f} {height - 34})">{cat}</text>'
        )

    legend_y = height - 14
    legend_x = margin["left"]
    for idx, item in enumerate(series):
        lx = legend_x + idx * 180
        parts.append(f'<rect x="{lx}" y="{legend_y - 10}" width="12" height="12" fill="{item["color"]}" rx="2" />')
        parts.append(f'<text x="{lx + 18}" y="{legend_y}" font-size="11" fill="#374151">{item["label"]}</text>')

    parts.append("</svg>")
    return "".join(parts)


def svg_simple_bars(
    labels: list[str],
    values: list[float],
    title: str,
    unit: str = "",
    width: int = 760,
    height: int = 320,
    color: str = "#bb4d34",
) -> str:
    margin = {"top": 34, "right": 24, "bottom": 38, "left": 180}
    inner_w = width - margin["left"] - margin["right"]
    row_h = inner_w  # dummy to silence lint-like reasoning
    row_h = (height - margin["top"] - margin["bottom"]) / max(len(labels), 1)
    max_v = max(values) if values else 1.0
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">',
        '<rect width="100%" height="100%" fill="white" rx="16" />',
        f'<text x="{margin["left"]}" y="20" font-size="16" font-weight="700" fill="#1f2937">{title}</text>',
    ]
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = margin["top"] + idx * row_h
        bar_w = inner_w * (value / max_v if max_v else 0)
        parts.append(f'<text x="{margin["left"] - 10}" y="{y + row_h*0.62:.1f}" text-anchor="end" font-size="12" fill="#374151">{label}</text>')
        parts.append(f'<rect x="{margin["left"]}" y="{y + row_h*0.18:.1f}" width="{bar_w:.1f}" height="{row_h*0.56:.1f}" fill="{color}" rx="5" />')
        parts.append(f'<text x="{margin["left"] + bar_w + 8:.1f}" y="{y + row_h*0.62:.1f}" font-size="11" fill="#6b7280">{value:.2f}{unit}</text>')
    parts.append("</svg>")
    return "".join(parts)


@dataclass
class Experiment:
    key: str
    title: str
    train_log: Path | None
    eval_path: Path | None
    scheme_metric_key: str | None
    notes: str


def extract_best_val(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    rows = load_jsonl(path)
    if not rows:
        return None
    return min(row["val_loss"] for row in rows if "val_loss" in row)


def load_eval(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return load_json_with_preamble(path)


def read_text_json(path: Path) -> dict:
    return json.load(path.open(encoding="utf-8"))


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def sample_block(title: str, prompt: str, text: str, plan: dict | None = None) -> str:
    plan_html = ""
    if plan:
        plan_items = ", ".join(f"{k}={v}" for k, v in plan.items())
        plan_html = f'<div class="sample-plan"><strong>План окончаний:</strong> {html_escape(plan_items)}</div>'
    return (
        '<div class="sample-card">'
        f'<h4>{html_escape(title)}</h4>'
        f'<div class="sample-prompt"><strong>Затравка:</strong> {html_escape(prompt)}</div>'
        f"{plan_html}"
        f'<pre>{html_escape(text)}</pre>'
        "</div>"
    )


def main() -> None:
    ARTICLE_DIR.mkdir(parents=True, exist_ok=True)

    processed_stats = read_text_json(ROOT / "data/processed/stats.json")
    processed_meta = read_text_json(ROOT / "data/processed/meta.json")
    rhyme_stats = read_text_json(ROOT / "data/processed_rhyme_focus/stats.json")
    rhyme_meta = read_text_json(ROOT / "data/processed_rhyme_focus/meta.json")
    aabb8_stats = read_text_json(ROOT / "data/processed_aabb8/stats.json")
    aabb8_meta = read_text_json(ROOT / "data/processed_aabb8/meta.json")
    aabb8_qf2_stats = read_text_json(ROOT / "data/processed_aabb8_qf2/stats.json")
    qwen_summary = read_text_json(ROOT / "artifacts/downloaded/vast_qwen3_8b_aabb_qf2_lora_bf16/summary.json")
    qwen_eval = read_text_json(ROOT / "artifacts/downloaded/vast_qwen3_8b_aabb_qf2_lora_bf16/eval8.json")

    experiments = [
        Experiment(
            key="aabb_baseline",
            title="AABB CCDD baseline",
            train_log=ROOT / "artifacts/checkpoints/host_5060_8line_20m/log.jsonl",
            eval_path=ROOT / "artifacts/checkpoints/host_5060_8line_20m/best.eval8.json",
            scheme_metric_key="aabb_ccdd_rate",
            notes="Основной 8-строчный baseline на quality-filtered корпусе.",
        ),
        Experiment(
            key="abab_branch",
            title="ABAB ABAB",
            train_log=ROOT / "artifacts/article_sync/host_5060_8line_abab_20m.log.jsonl",
            eval_path=ROOT / "artifacts/article_sync/host_5060_8line_abab_20m.best.eval8.json",
            scheme_metric_key="abab_abab_rate",
            notes="Альтернативная схема с большей частотой в корпусе.",
        ),
        Experiment(
            key="staged",
            title="Stage1 → Stage2",
            train_log=ROOT / "artifacts/article_sync/host_5060_aabb_qf2_stage2_from_fullpoem_20m.log.jsonl",
            eval_path=ROOT / "artifacts/article_sync/host_5060_aabb_qf2_stage2_from_fullpoem_20m.best.eval8.json",
            scheme_metric_key="aabb_ccdd_rate",
            notes="Предобучение на полных стихах с последующим дообучением на строгой задаче.",
        ),
        Experiment(
            key="planner",
            title="Planner-guided",
            train_log=ROOT / "artifacts/article_sync/host_5060_aabb_with_plan_20m.log.jsonl",
            eval_path=ROOT / "artifacts/article_sync/host_5060_aabb_with_plan_20m.best.planned.eval8.json",
            scheme_metric_key="aabb_ccdd_rate",
            notes="Двухшаговая архитектура: предсказание окончаний строк 2/4/6/8 и последующая генерация.",
        ),
    ]

    exp_rows = []
    for exp in experiments:
        ev = load_eval(exp.eval_path)
        scheme_value = ev.get(exp.scheme_metric_key) if ev and exp.scheme_metric_key else None
        exp_rows.append(
            {
                "title": exp.title,
                "best_val": extract_best_val(exp.train_log),
                "exact_8": ev.get("exact_8_lines_rate") if ev else None,
                "second_rhyme": ev.get("second_line_rhyme_rate") if ev else None,
                "scheme": scheme_value,
                "notes": exp.notes,
            }
        )

    prehistory_rows = [
        {
            "title": "Начальный baseline (без 8-строчной постановки)",
            "dataset": "processed",
            "best_val": 3.2320,
            "second_rhyme": 0.12,
            "comment": "Генерация длиннее, но рифма слабая.",
        },
        {
            "title": "Rhyme-focused retrain",
            "dataset": "processed_rhyme_focus",
            "best_val": 2.9590,
            "second_rhyme": 0.55,
            "comment": "Рифма резко улучшилась, но генерация схлопнулась к 2 строкам.",
        },
    ]

    loss_svg = svg_line_chart(
        [
            {
                "label": "AABB baseline",
                "color": "#bb4d34",
                "points": [(r["iter"], r["val_loss"]) for r in load_jsonl(ROOT / "artifacts/checkpoints/host_5060_8line_20m/log.jsonl")],
            },
            {
                "label": "ABAB",
                "color": "#2f6f73",
                "points": [(r["iter"], r["val_loss"]) for r in load_jsonl(ROOT / "artifacts/article_sync/host_5060_8line_abab_20m.log.jsonl")],
            },
            {
                "label": "Stage2",
                "color": "#5b8c49",
                "points": [(r["iter"], r["val_loss"]) for r in load_jsonl(ROOT / "artifacts/article_sync/host_5060_aabb_qf2_stage2_from_fullpoem_20m.log.jsonl")],
            },
            {
                "label": "Planner-guided",
                "color": "#6b4f9d",
                "points": [(r["iter"], r["val_loss"]) for r in load_jsonl(ROOT / "artifacts/article_sync/host_5060_aabb_with_plan_20m.log.jsonl")],
            },
        ],
        "Кривые валидационной потери для основных 8-строчных веток",
    )

    planner_loss_svg = svg_line_chart(
        [
            {
                "label": "Stage1 full poems",
                "color": "#7c3f58",
                "points": [(r["iter"], r["val_loss"]) for r in load_jsonl(ROOT / "artifacts/article_sync/host_5060_fullpoem_20m_stage1.log.jsonl")],
            },
            {
                "label": "Planner endings",
                "color": "#2563eb",
                "points": [(r["iter"], r["val_loss"]) for r in load_jsonl(ROOT / "artifacts/article_sync/host_5060_aabb_end_planner_12m.log.jsonl")],
            },
        ],
        "Вспомогательные ветки: предобучение на полных стихах и planner окончаний",
    )

    metric_svg = svg_grouped_bar_chart(
        categories=[row["title"] for row in exp_rows],
        series=[
            {
                "label": "exact_8_lines_rate",
                "color": "#bb4d34",
                "values": [row["exact_8"] or 0.0 for row in exp_rows],
            },
            {
                "label": "second_line_rhyme_rate",
                "color": "#2f6f73",
                "values": [row["second_rhyme"] or 0.0 for row in exp_rows],
            },
            {
                "label": "scheme_rate",
                "color": "#6b4f9d",
                "values": [row["scheme"] or 0.0 for row in exp_rows],
            },
        ],
        title="Сравнение качества 8-строчных моделей",
    )

    baseline_eval = load_eval(ROOT / "artifacts/checkpoints/host_5060_8line_20m/best.eval8.json")
    abab_eval = load_eval(ROOT / "artifacts/article_sync/host_5060_8line_abab_20m.best.eval8.json")
    staged_eval = load_eval(ROOT / "artifacts/article_sync/host_5060_aabb_qf2_stage2_from_fullpoem_20m.best.eval8.json")
    planned_eval = load_eval(ROOT / "artifacts/article_sync/host_5060_aabb_with_plan_20m.best.planned.eval8.json")

    base_compare_svg = svg_grouped_bar_chart(
        categories=["Scratch AABB CCDD", "Qwen3-8B-Base + LoRA"],
        series=[
            {
                "label": "exact_8_lines_rate",
                "color": "#bb4d34",
                "values": [baseline_eval["exact_8_lines_rate"], qwen_eval["exact_8_lines_rate"]],
            },
            {
                "label": "second_line_rhyme_rate",
                "color": "#2f6f73",
                "values": [baseline_eval["second_line_rhyme_rate"], qwen_eval["second_line_rhyme_rate"]],
            },
            {
                "label": "scheme_rate",
                "color": "#6b4f9d",
                "values": [baseline_eval["aabb_ccdd_rate"], qwen_eval["aabb_ccdd_rate"]],
            },
        ],
        title="Scratch baseline против base-модели Qwen3-8B-Base",
    )

    corpus_svg = svg_simple_bars(
        labels=[
            "processed: rows_kept",
            "processed_rhyme_focus: rows_written",
            "processed_aabb8: windows_kept",
            "processed_aabb8_qf2: rows_kept",
        ],
        values=[
            float(processed_stats["rows_kept"]),
            float(rhyme_stats["train_rows_written"] + rhyme_stats["val_rows_written"] + rhyme_stats["test_rows_written"]),
            float(aabb8_stats["train_windows_kept"] + aabb8_stats["val_windows_kept"] + aabb8_stats["test_windows_kept"]),
            float(aabb8_qf2_stats["train_rows_kept"] + aabb8_qf2_stats["val_rows_kept"] + aabb8_qf2_stats["test_rows_kept"]),
        ],
        title="Сжатие корпуса при переходе к более строгой постановке",
    )

    examples_html = "".join(
        [
            sample_block("Baseline AABB", baseline_eval["preview"][0]["prompt"], baseline_eval["preview"][0]["generated"]),
            sample_block("ABAB", abab_eval["preview"][0]["prompt"], abab_eval["preview"][0]["generated"]),
            sample_block("Stage1 → Stage2", staged_eval["preview"][0]["prompt"], staged_eval["preview"][0]["generated"]),
            sample_block(
                "Planner-guided",
                planned_eval["preview"][2]["prompt"],
                planned_eval["preview"][2]["generated"],
                planned_eval["preview"][2]["plan"],
            ),
            sample_block("Qwen3-8B-Base + LoRA", qwen_eval["preview"][0]["prompt"], qwen_eval["preview"][0]["generated"]),
        ]
    )

    conclusions = [
        "Обучение с нуля только на корпусе русской поэзии позволяет надежно навязать формальные ограничения длины и рифмовки, но не обеспечивает естественность языка на уровне, сопоставимом с общим языковым pretraining.",
        "Наиболее успешной оказалась узкая постановка AABB CCDD на quality-filtered корпусе: она дала почти идеальную длину в 8 строк и лучший баланс между рифмой и структурой.",
        "Более частая в корпусе схема ABAB ABAB не дала улучшения: малой decoder-only модели оказалось трудно удерживать рифму через строку.",
        "Предобучение на полных стихах улучшило next-token objective, но ухудшило целевую constrained-generation задачу; этот отрицательный результат важен сам по себе.",
        "Planner-guided архитектура оказалась лучше staged-подхода, но не превзошла строгий baseline, что указывает на недостаточную силу planner-а при отсутствии общего языкового фундамента.",
        "Контринтуитивно, сильная base-модель Qwen3-8B-Base после LoRA-дообучения не превзошла scratch-baseline: она почти идеально удерживала длину, но почти полностью провалила рифму и схему AABB CCDD.",
        "Главное ограничение эксперимента состоит в том, что модель никогда не видела большого общего корпуса русского языка; именно это, вероятно, является основной причиной слабой семантики и появления псевдослов.",
    ]

    template = Template(
        """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Обучение с нуля русской поэтической модели продолжения стихов</title>
  <style>
    :root {
      --bg: #f6f1e8;
      --paper: #fffdf8;
      --ink: #1f2937;
      --muted: #5b6473;
      --edge: #ddd1c3;
      --accent: #9f3925;
      --accent-2: #2f6f73;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #eee4d6 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      line-height: 1.55;
    }
    main {
      max-width: 1080px;
      margin: 0 auto;
      padding: 36px 20px 72px;
    }
    .paper {
      background: var(--paper);
      border: 1px solid var(--edge);
      border-radius: 24px;
      box-shadow: 0 20px 45px rgba(43, 31, 20, 0.08);
      padding: 34px 34px 44px;
    }
    h1, h2, h3, h4 { line-height: 1.15; color: #18212c; }
    h1 { font-size: 2.45rem; margin: 0 0 14px; letter-spacing: -0.03em; }
    h2 { margin-top: 38px; font-size: 1.55rem; }
    h3 { margin-top: 28px; font-size: 1.18rem; }
    p, li { font-size: 1.03rem; }
    .lead { font-size: 1.12rem; color: var(--muted); max-width: 900px; }
    .meta {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin: 22px 0 10px;
    }
    .meta-card, .box {
      background: #fffaf2;
      border: 1px solid var(--edge);
      border-radius: 18px;
      padding: 14px 16px;
    }
    .meta-card strong { display: block; margin-bottom: 6px; font-size: 0.9rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0 10px;
      font-size: 0.98rem;
    }
    th, td {
      border-bottom: 1px solid #eadfd2;
      padding: 10px 10px;
      vertical-align: top;
      text-align: left;
    }
    th { color: var(--muted); font-weight: 700; }
    .figure {
      margin: 22px 0;
      padding: 14px;
      border: 1px solid var(--edge);
      border-radius: 18px;
      background: #fff;
    }
    .caption {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.94rem;
    }
    .sample-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 12px;
    }
    .sample-card {
      border: 1px solid var(--edge);
      border-radius: 18px;
      padding: 14px;
      background: #fffcf7;
    }
    .sample-card h4 { margin: 0 0 8px; }
    .sample-card pre {
      margin: 10px 0 0;
      white-space: pre-wrap;
      font-family: "IBM Plex Mono", "Fira Code", monospace;
      font-size: 0.92rem;
      line-height: 1.45;
      color: #1d2430;
    }
    .sample-prompt, .sample-plan { color: var(--muted); font-size: 0.95rem; }
    .conclusion-list li { margin-bottom: 8px; }
    .small { color: var(--muted); font-size: .92rem; }
    code { font-family: "IBM Plex Mono", "Fira Code", monospace; font-size: 0.93em; }
    @media (max-width: 860px) {
      .meta, .sample-grid { grid-template-columns: 1fr; }
      .paper { padding: 22px 18px 32px; }
      h1 { font-size: 2rem; }
    }
  </style>
</head>
<body>
  <main>
    <article class="paper">
      <h1>Обучение с нуля русской поэтической модели продолжения стихов по первой строке</h1>
      <p class="lead">
        В работе исследуется, насколько далеко можно продвинуться в задаче продолжения русскоязычного стихотворения,
        если жестко запретить использование готовой базовой языковой модели и обучать decoder-only Transformer с нуля
        только на поэтическом корпусе. Основной фокус эксперимента — конфликт между формальными ограничениями
        (длина, рифма, схема строфы) и семантической/языковой естественностью. На финальном этапе этот scratch-подход
        был также сопоставлен с сильной base-моделью <code>Qwen3-8B-Base</code>, дообученной на том же корпусе.
      </p>

      <div class="meta">
        <div class="meta-card"><strong>Язык</strong>Русский</div>
        <div class="meta-card"><strong>Вход</strong>Одна первая строка</div>
        <div class="meta-card"><strong>Целевая форма</strong>8 строк, схема <code>AABB CCDD</code></div>
      </div>

      <h2>Аннотация</h2>
      <p>
        Был построен полный экспериментальный цикл: извлечение и очистка корпуса русской поэзии, разметка рифмовки
        и ударений, обучение нескольких генеративных моделей с нуля, а также сравнение альтернативных постановок задачи.
        Показано, что формальные свойства стиха частично поддаются обучению даже при сравнительно малом размере модели,
        однако отсутствие общего языкового pretraining существенно ограничивает связность, словарь и естественность текста.
        Наилучший результат по совокупности формальных метрик был достигнут узкой 8-строчной постановкой <code>AABB CCDD</code>
        на quality-filtered корпусе. Более частая схема <code>ABAB ABAB</code>, staged-подход с предобучением на полных стихах и
        двухшаговая planner-guided архитектура не улучшили baseline. Дополнительное сравнение с <code>Qwen3-8B-Base</code>
        показало, что сильная base-модель после LoRA-дообучения лучше удерживает общую языковую форму, но без специального
        objective не усваивает строгую рифмованную схему и проигрывает scratch-baseline по целевой метрике.
      </p>

      <h2>1. Постановка задачи</h2>
      <p>
        Исходная цель проекта заключалась в проверке следующей гипотезы: можно ли обучить небольшую нейросетевую модель с нуля,
        используя только русскоязычный поэтический корпус, так чтобы по одной строке она продолжала стихотворение в формально
        заданной структуре. Принципиальное ограничение эксперимента состояло в отказе от любых готовых русских или многоязычных
        базовых языковых моделей.
      </p>
      <p>
        На раннем этапе была проверена свободная постановка задачи продолжения стиха, после чего эксперимент был сужен до
        более контролируемого сценария: входом служит одна строка, выходом — стихотворение ровно из восьми строк с
        рифмовкой <code>AABB CCDD</code>. Именно эта постановка оказалась наиболее удобной для количественной оценки и
        для сравнения архитектурных и корпусных решений.
      </p>

      <h2>2. Данные и подготовка корпуса</h2>
      <h3>2.1. Исходный корпус</h3>
      <p>
        Базой послужил большой русскоязычный поэтический корпус, из которого после первичной фильтрации было оставлено
        {{ processed_rows }} текстов. На этапе грубой эвристической разметки среди них удалось выделить
        {{ scheme_abab }} текстов с quatrain-схемой <code>ABAB</code>, {{ scheme_aabb }} — со схемой <code>AABB</code>
        и {{ scheme_abba }} — со схемой <code>ABBA</code>.
      </p>
      <div class="box small">
        Первичный тренировочный массив содержал {{ processed_train_tokens }} токенов, что достаточно для базового
        специализированного language modeling, но существенно меньше типичных объемов общего языкового pretraining.
      </div>

      <div class="figure">
        {{ corpus_svg | safe }}
        <div class="caption">
          Рисунок 1. Уменьшение объема доступных данных при переходе от общего поэтического корпуса к строгой постановке
          <code>AABB CCDD</code> и дальнейшей quality-фильтрации.
        </div>
      </div>

      <h3>2.2. Этапы сужения корпуса</h3>
      <table>
        <thead>
          <tr>
            <th>Корпус</th>
            <th>Назначение</th>
            <th>Размер</th>
            <th>Комментарий</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>processed</code></td>
            <td>Общий baseline на поэзии</td>
            <td>{{ processed_rows }}</td>
            <td>Широкий корпус с разметкой quatrain-схем.</td>
          </tr>
          <tr>
            <td><code>processed_rhyme_focus</code></td>
            <td>Усиление рифмы второй строки</td>
            <td>{{ rhyme_rows }}</td>
            <td>Только <code>AABB</code> + oversampling первых двух строк.</td>
          </tr>
          <tr>
            <td><code>processed_aabb8</code></td>
            <td>Строгая 8-строчная задача</td>
            <td>{{ aabb8_rows }}</td>
            <td>8-строчные окна с проверенной схемой <code>AABB CCDD</code>.</td>
          </tr>
          <tr>
            <td><code>processed_aabb8_qf2</code></td>
            <td>Основной корпус финального baseline</td>
            <td>{{ aabb8_qf2_rows }}</td>
            <td>Дополнительная quality-фильтрация: дубликаты, дешевые рифмы, разрывные фрагменты, сбои метра.</td>
          </tr>
        </tbody>
      </table>

      <p>
        Quality filter v2 был нацелен не на литературную оценку текста, а на снижение task-specific шума: убирались
        окна с явными повторными строками, одинаковыми последними словами в рифмующихся парах, фрагментными концовками,
        сильными межпарными рифмовыми коллизиями и грубыми просодическими перекосами.
      </p>

      <h2>3. Архитектура и схема обучения</h2>
      <p>
        Во всех основных ветках использовалась одна базовая архитектура: <code>decoder-only Transformer</code> с обучением с нуля.
        Для основной 8-строчной модели конфигурация составляла 8 слоев самовнимания, 6 attention heads, скрытую размерность 384,
        контекст 256 токенов и словарь SentencePiece unigram на 16 000 токенов. Структурные токены задавали длину,
        схему строфы и позицию строки внутри восьмистрочника.
      </p>
      <p>
        Были протестированы четыре основные ветки: (1) строгий baseline <code>AABB CCDD</code>, (2) альтернативная схема
        <code>ABAB ABAB</code>, (3) staged-обучение через предобучение на полных стихах и последующее дообучение на
        строгой задаче, (4) planner-guided архитектура, в которой сначала предсказываются окончания строк 2/4/6/8,
        а затем генератор строит весь восьмистрочник под этот план.
      </p>

      <div class="figure">
        {{ loss_svg | safe }}
        <div class="caption">
          Рисунок 2. Сопоставление кривых валидационной потери для основных 8-строчных веток.
        </div>
      </div>

      <div class="figure">
        {{ planner_loss_svg | safe }}
        <div class="caption">
          Рисунок 3. Вспомогательные ветки: stage-1 предобучение на полных стихах и отдельный planner окончаний.
        </div>
      </div>

      <h2>4. Методы оценки</h2>
      <p>
        Для оценки использовались как loss-based метрики, так и task-specific критерии. Для финальной восьмистрочной задачи
        основными считались три показателя: <code>exact_8_lines_rate</code> (генерация ровно восьми строк),
        <code>second_line_rhyme_rate</code> (рифма между первой и второй строкой) и <code>scheme_rate</code>
        (соблюдение полной целевой схемы строфы). Для planner-а дополнительно измерялись точное совпадение конечных слов
        и совпадение ударного рифменного хвоста.
      </p>

      <div class="figure">
        {{ metric_svg | safe }}
        <div class="caption">
          Рисунок 4. Сравнение качества четырех 8-строчных моделей по ключевым task-specific метрикам.
        </div>
      </div>

      <h2>5. Результаты</h2>
      <h3>5.1. Предварительные эксперименты</h3>
      <table>
        <thead>
          <tr>
            <th>Эксперимент</th>
            <th>Корпус</th>
            <th>Лучший val_loss</th>
            <th>Рифма 1–2</th>
            <th>Наблюдение</th>
          </tr>
        </thead>
        <tbody>
          {% for row in prehistory_rows %}
          <tr>
            <td>{{ row.title }}</td>
            <td><code>{{ row.dataset }}</code></td>
            <td>{{ "%.4f"|format(row.best_val) }}</td>
            <td>{{ "%.2f"|format(row.second_rhyme) }}</td>
            <td>{{ row.comment }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>

      <p>
        Эти ранние результаты были важны для уточнения постановки задачи. Они показали, что рифму между первой и второй
        строкой можно существенно улучшить, однако без явного контроля длины и структуры генерация быстро схлопывается
        к очень коротким ответам.
      </p>

      <h3>5.2. Основные 8-строчные ветки</h3>
      <table>
        <thead>
          <tr>
            <th>Модель</th>
            <th>Лучший val_loss</th>
            <th>Ровно 8 строк</th>
            <th>Рифма 1–2</th>
            <th>Полная схема</th>
            <th>Комментарий</th>
          </tr>
        </thead>
        <tbody>
          {% for row in exp_rows %}
          <tr>
            <td>{{ row.title }}</td>
            <td>{{ "%.4f"|format(row.best_val) }}</td>
            <td>{{ "%.4f"|format(row.exact_8) }}</td>
            <td>{{ "%.4f"|format(row.second_rhyme) }}</td>
            <td>{{ "%.4f"|format(row.scheme) }}</td>
            <td>{{ row.notes }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>

      <p>
        Наиболее сильной моделью по совокупности формальных ограничений оказался baseline <code>AABB CCDD</code>.
        Он почти идеально удерживал длину в восемь строк и давал наилучшую полную схему рифмовки. Ветка
        <code>ABAB ABAB</code>, несмотря на более высокую частоту в корпусе, оказалась хуже; это указывает на то,
        что для малой модели рифма через строку оказывается сложнее, чем более локальная схема <code>AABB</code>.
      </p>
      <p>
        Staged-подход интересен тем, что улучшил loss, но ухудшил именно целевую constrained-generation задачу.
        Это означает, что более хороший next-token objective на близком домене не гарантирует лучшей управляемой
        генерации в узкой формальной постановке. Planner-guided архитектура, в свою очередь, превзошла staged-ветку,
        однако также не обошла baseline.
      </p>

      <h3>5.3. Сравнение со strong base model</h3>
      <p>
        После завершения scratch-линейки был проведен отдельный контрольный эксперимент с <code>Qwen3-8B-Base</code>,
        дообученной методом LoRA на том же quality-filtered корпусе и под ту же постановку <code>AABB CCDD</code>.
        Этот шаг был важен для проверки естественной гипотезы: не является ли основной предел scratch-подхода просто
        отсутствием большого языкового фундамента.
      </p>

      <div class="figure">
        {{ base_compare_svg | safe }}
        <div class="caption">
          Рисунок 5. Формальное качество лучшего scratch-baseline и <code>Qwen3-8B-Base</code> после LoRA-дообучения.
        </div>
      </div>

      <table>
        <thead>
          <tr>
            <th>Модель</th>
            <th>Лучший eval_loss</th>
            <th>Ровно 8 строк</th>
            <th>Рифма 1–2</th>
            <th>Полная схема AABB CCDD</th>
            <th>Интерпретация</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Scratch AABB CCDD</td>
            <td>{{ "%.4f"|format(exp_rows[0].best_val) }}</td>
            <td>{{ "%.4f"|format(baseline_eval["exact_8_lines_rate"]) }}</td>
            <td>{{ "%.4f"|format(baseline_eval["second_line_rhyme_rate"]) }}</td>
            <td>{{ "%.4f"|format(baseline_eval["aabb_ccdd_rate"]) }}</td>
            <td>Лучшая модель по формальным ограничениям внутри scratch-эксперимента.</td>
          </tr>
          <tr>
            <td>Qwen3-8B-Base + LoRA</td>
            <td>{{ "%.4f"|format(qwen_best_eval_loss) }}</td>
            <td>{{ "%.4f"|format(qwen_eval["exact_8_lines_rate"]) }}</td>
            <td>{{ "%.4f"|format(qwen_eval["second_line_rhyme_rate"]) }}</td>
            <td>{{ "%.4f"|format(qwen_eval["aabb_ccdd_rate"]) }}</td>
            <td>Почти идеально удерживает длину, но практически не усваивает рифмовую схему в текущем формате SFT.</td>
          </tr>
        </tbody>
      </table>

      <p>
        Этот результат оказался контринтуитивным, но методологически очень важным. В нашей постановке сильная base-модель
        проиграла узкой scratch-модели именно по целевой задаче. Иными словами, общий языковой prior сам по себе не
        гарантирует успех, если objective не заставляет модель считать рифму и строфическую схему обязательными.
        Scratch-baseline хуже по общей языковой культуре, но лучше специализирован на локальной рифмованной форме.
      </p>

      <h3>5.4. Качественный анализ генераций</h3>
      <p>
        Формальные метрики не исчерпывают качества поэтического текста. Ниже приведены характерные примеры генерации.
        Видно, что даже лучшая модель нередко удерживает схему и длину ценой семантической рыхлости, а planner-guided
        ветка в текущем виде способна навязывать структуру, но иногда генерирует псевдослова и распадающийся синтаксис.
        Генерация <code>Qwen3-8B-Base</code> демонстрирует обратную картину: длина удерживается уверенно, но сама рифма
        и целевая схема почти не соблюдаются.
      </p>
      <div class="sample-grid">
        {{ examples_html | safe }}
      </div>

      <h2>6. Обсуждение</h2>
      <p>
        Центральный вывод эксперимента состоит в том, что обучение с нуля только на поэтическом корпусе позволяет
        частично овладеть формой, но почти неизбежно страдает по смыслу и языковой естественности. Поэтический корпус,
        даже достаточно большой в абсолютных числах, остается слишком малым и слишком специализированным, чтобы заменить
        общий языковой pretraining. Модель учится рифмовать, завершать строки и имитировать поэтическую поверхность,
        но не получает достаточного фундамента для устойчивой семантической композиции.
      </p>
      <p>
        Одновременно сравнение с <code>Qwen3-8B-Base</code> показывает, что и обратный тезис в простой форме неверен:
        наличие сильного языкового фундамента еще не означает автоматического решения задачи. При стандартном completion-style
        SFT base-модель охотно пишет восьмистрочные тексты, но не воспринимает рифму как жесткое ограничение. Следовательно,
        дальнейшее движение должно идти не только в сторону более сильной базы, но и в сторону более подходящего обучающего
        сигнала для поэтической формы.
      </p>
      <p>
        Planner-guided архитектура была введена именно для ослабления нагрузки на генератор: предполагалось, что если
        вынести планирование окончаний в отдельную модель, основной генератор сможет сосредоточиться на содержании строк.
        На практике planner оказался слишком слаб, чтобы надежно прогнозировать будущие окончания, и downstream-выигрыш
        не перекрыл потери относительно простого strict baseline. Тем не менее отрицательный результат здесь полезен:
        он показывает, что сама идея декомпозиции перспективна, но без более сильного языкового фундамента или более
        качественного planner-а она не дает преимущества.
      </p>

      <h2>7. Угрозы валидности</h2>
      <ul>
        <li>Корпус неоднороден по качеству и стилю; в нем присутствуют как сильные тексты, так и любительская поэзия.</li>
        <li>Часть рифмовой и просодической разметки основана на эвристиках и стресс-модели, а не на полностью ручной аннотации.</li>
        <li>Сравнение моделей проводилось в рамках ограниченного бюджета вычислений и на относительно малой архитектуре.</li>
        <li>Оценка осмысленности остается преимущественно качественной; формальные метрики не отражают глубину поэтического содержания.</li>
      </ul>

      <h2>8. Заключение</h2>
      <ul class="conclusion-list">
        {% for item in conclusions %}
        <li>{{ item }}</li>
        {% endfor %}
      </ul>

      <h2>9. Практический вывод</h2>
      <p>
        Если целью является именно <em>эксперимент</em> по обучению с нуля на поэзии, то проект достиг содержательного
        результата: удалось построить несколько сравнимых веток, показать обучаемость формальных ограничений и получить
        серию как положительных, так и отрицательных эмпирических выводов. Если же цель смещается к качественной генерации
        осмысленных стихов, то следующий шаг почти неизбежно связан либо с большим общим русским корпусом для pretraining,
        либо с использованием уже готовой базовой языковой модели.
      </p>

      <p class="small">
        Статья автоматически собрана из <code>HANDOFF.md</code>, локальных логов обучения, quality-filtered статистик
        и удаленных артефактов GPU-хоста, синхронизированных для анализа.
      </p>
    </article>
  </main>
</body>
</html>
        """
    )

    html = template.render(
        processed_rows=format_int(processed_stats["rows_kept"]),
        processed_train_tokens=format_int(processed_meta["train"]),
        scheme_abab=format_int(processed_stats["scheme_ABAB"]),
        scheme_aabb=format_int(processed_stats["scheme_AABB"]),
        scheme_abba=format_int(processed_stats["scheme_ABBA"]),
        rhyme_rows=format_int(
            rhyme_stats["train_rows_written"] + rhyme_stats["val_rows_written"] + rhyme_stats["test_rows_written"]
        ),
        aabb8_rows=format_int(
            aabb8_stats["train_windows_kept"] + aabb8_stats["val_windows_kept"] + aabb8_stats["test_windows_kept"]
        ),
        aabb8_qf2_rows=format_int(
            aabb8_qf2_stats["train_rows_kept"] + aabb8_qf2_stats["val_rows_kept"] + aabb8_qf2_stats["test_rows_kept"]
        ),
        loss_svg=loss_svg,
        planner_loss_svg=planner_loss_svg,
        metric_svg=metric_svg,
        base_compare_svg=base_compare_svg,
        corpus_svg=corpus_svg,
        prehistory_rows=prehistory_rows,
        exp_rows=exp_rows,
        examples_html=examples_html,
        conclusions=conclusions,
        qwen_best_eval_loss=qwen_summary["best_metric"],
        baseline_eval=baseline_eval,
        qwen_eval=qwen_eval,
    )

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(OUTPUT_HTML)


if __name__ == "__main__":
    main()
