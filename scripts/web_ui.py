from __future__ import annotations

import argparse
import html
import subprocess

import gradio as gr

from poetry_lm.inference import (
    generate_text,
    load_bundle,
    resolve_device,
)
from poetry_lm.model_registry import model_specs


MAX_NEW_TOKENS = 160


def training_is_running() -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-af", r"python scripts/train.py"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return bool(result.stdout.strip())


def _status_badge(text: str, ok: bool) -> str:
    bg = "#d7f5dc" if ok else "#f7d9d6"
    fg = "#14532d" if ok else "#7f1d1d"
    return (
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{bg};color:{fg};font-weight:600'>{html.escape(text)}</span>"
    )


def model_status_markdown(device: str) -> str:
    rows: list[str] = []
    for spec in model_specs():
        checkpoint_exists = all(path.exists() for path in spec.all_checkpoints())
        tokenizer_exists = all(path.exists() for path in spec.all_tokenizers())
        ok = checkpoint_exists and tokenizer_exists
        status = _status_badge("ready", True) if ok else _status_badge("pending", False)
        checkpoint_text = "<br>".join(
            html.escape(str(path)) for path in spec.all_checkpoints()
        )
        tokenizer_text = "<br>".join(html.escape(str(path)) for path in spec.all_tokenizers())
        note = html.escape(spec.note)
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(spec.title)}</strong><br><span style='color:#555'>{note}</span></td>"
            f"<td>{status}</td>"
            f"<td><code>{checkpoint_text}</code></td>"
            f"<td><code>{tokenizer_text}</code></td>"
            "</tr>"
        )

    return (
        f"### Состояние моделей\n"
        f"<div style='margin:6px 0 12px;color:#555'>Device: <code>{html.escape(device)}</code>, "
        f"max_new_tokens: <code>{MAX_NEW_TOKENS}</code></div>"
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr>"
        "<th align='left'>Модель</th>"
        "<th align='left'>Статус</th>"
        "<th align='left'>Checkpoint</th>"
        "<th align='left'>Tokenizer</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def generate_all(prompt: str, temperature: float, top_k: int, device: str):
    prompt = prompt.strip()
    if not prompt:
        raise gr.Error("Нужна первая строка.")

    outputs: list[str] = []
    for spec in model_specs():
        if not all(path.exists() for path in spec.all_checkpoints()):
            outputs.append("Чекпойнт еще не готов.")
            continue
        if not all(path.exists() for path in spec.all_tokenizers()):
            outputs.append("Tokenizer не найден.")
            continue
        try:
            if spec.is_planner_guided:
                from poetry_lm.inference import generate_text_with_planner

                planner_bundle = load_bundle(spec.planner_checkpoint, spec.tokenizer, device=device)
                generator_bundle = load_bundle(spec.checkpoint, spec.tokenizer, device=device)
                _, output = generate_text_with_planner(
                    planner_bundle=planner_bundle,
                    generator_bundle=generator_bundle,
                    prompt=prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=float(temperature),
                    top_k=int(top_k),
                )
                outputs.append(output)
            else:
                bundle = load_bundle(spec.checkpoint, spec.tokenizer, device=device)
                outputs.append(
                    generate_text(
                        bundle=bundle,
                        prompt=prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=float(temperature),
                        top_k=int(top_k),
                    )
                )
        except Exception as exc:  # pragma: no cover - UI runtime safety
            outputs.append(f"Ошибка инференса: {exc}")

    return (model_status_markdown(device), *outputs)


def build_demo(device: str) -> gr.Blocks:
    css = """
    .poetry-shell {max-width: 1480px; margin: 0 auto;}
    .poetry-note {color: #555; font-size: 0.95rem;}
    .poetry-out textarea {font-family: 'IBM Plex Mono', 'Fira Code', monospace;}
    """
    with gr.Blocks(css=css, title="Poetry LM Compare") as demo:
        device_state = gr.State(device)
        gr.Markdown(
            """
            # Poetry LM Compare
            Текущий основной baseline по одной первой строке.
            """,
            elem_classes=["poetry-shell"],
        )
        with gr.Row():
            prompt = gr.Textbox(
                label="Первая строка",
                placeholder="Жизнь просит, требует словами,",
                lines=2,
                max_lines=2,
                scale=6,
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.9,
                step=0.05,
                label="Temperature",
                scale=2,
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=100,
                value=50,
                step=1,
                label="Top-k",
                scale=2,
            )
        with gr.Row():
            generate_btn = gr.Button("Сгенерировать", variant="primary")
            refresh_btn = gr.Button("Обновить статусы", variant="secondary")

        status = gr.Markdown(model_status_markdown(device))

        with gr.Row():
            output_aabb = gr.Textbox(
                label="AABB CCDD baseline",
                lines=18,
                max_lines=24,
                elem_classes=["poetry-out"],
                buttons=["copy"],
            )

        gr.Markdown(
            "Пока скрытое по умолчанию: `max_new_tokens=160`, device выбирается автоматически. "
            "Если позже понадобится воспроизводимость, добавим `seed`.",
            elem_classes=["poetry-note"],
        )

        outputs = [status, output_aabb]
        generate_btn.click(
            fn=generate_all,
            inputs=[prompt, temperature, top_k, device_state],
            outputs=outputs,
            concurrency_limit=1,
        )
        prompt.submit(
            fn=generate_all,
            inputs=[prompt, temperature, top_k, device_state],
            outputs=outputs,
            concurrency_limit=1,
        )
        refresh_btn.click(
            fn=model_status_markdown,
            inputs=[device_state],
            outputs=[status],
            concurrency_limit=1,
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    requested_device = args.device
    if requested_device == "auto" and training_is_running():
        device = "cpu"
    else:
        device = resolve_device(requested_device)
    demo = build_demo(device)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        show_api=False,
    )


if __name__ == "__main__":
    main()
