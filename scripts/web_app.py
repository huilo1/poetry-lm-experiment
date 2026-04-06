from __future__ import annotations

import argparse
import os
from textwrap import dedent

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn


DEFAULT_INFERENCE_API_BASE_URL = os.environ.get("INFERENCE_API_BASE_URL", "http://127.0.0.1:8001")


class GenerateRequest(BaseModel):
    model_key: str
    prompt: str = Field(min_length=1, max_length=400)
    temperature: float = Field(default=0.9, ge=0.1, le=1.5)
    top_k: int = Field(default=50, ge=1, le=100)


def build_index_html() -> str:
    return dedent(
        """
        <!doctype html>
        <html lang="ru">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Poetry LM Compare</title>
          <style>
            :root {
              --bg: #f2eadf;
              --paper: #fffaf1;
              --ink: #221b18;
              --muted: #685c55;
              --edge: #d8c8bc;
              --accent: #bb4d34;
              --accent-soft: #f5d4c5;
            }
            * { box-sizing: border-box; }
            body {
              margin: 0;
              background:
                radial-gradient(circle at top left, #fff7ee 0, transparent 28%),
                linear-gradient(180deg, #efe4d4 0%, var(--bg) 100%);
              color: var(--ink);
              font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
            }
            .shell {
              max-width: 1520px;
              margin: 0 auto;
              padding: 28px 20px 36px;
            }
            h1 {
              margin: 0 0 8px;
              font-size: clamp(2rem, 4vw, 3.6rem);
              line-height: 0.95;
              letter-spacing: -0.04em;
            }
            .lead {
              max-width: 920px;
              margin: 0 0 20px;
              color: var(--muted);
              font-size: 1.05rem;
            }
            .panel, .card {
              background: color-mix(in srgb, var(--paper) 92%, white);
              border: 1px solid var(--edge);
              border-radius: 22px;
              box-shadow: 0 14px 30px rgba(65, 42, 29, 0.06);
            }
            .panel { padding: 20px; margin-bottom: 18px; }
            .controls {
              display: grid;
              grid-template-columns: 1.9fr 0.8fr 0.8fr auto auto;
              gap: 14px;
              align-items: end;
            }
            label {
              display: block;
              margin-bottom: 8px;
              font-size: 0.88rem;
              color: var(--muted);
              text-transform: uppercase;
              letter-spacing: 0.08em;
            }
            textarea, input {
              width: 100%;
              border: 1px solid var(--edge);
              border-radius: 16px;
              background: #fffdfa;
              color: var(--ink);
              padding: 14px 16px;
              font: inherit;
            }
            textarea {
              min-height: 88px;
              resize: vertical;
            }
            input[type="range"] {
              padding: 0;
              accent-color: var(--accent);
            }
            .range-meta {
              display: flex;
              justify-content: space-between;
              color: var(--muted);
              font-size: 0.9rem;
              margin-top: 8px;
            }
            button {
              appearance: none;
              border: 0;
              border-radius: 16px;
              padding: 14px 18px;
              font: inherit;
              cursor: pointer;
              transition: transform .12s ease, opacity .12s ease;
            }
            button:hover { transform: translateY(-1px); }
            .primary {
              background: var(--accent);
              color: #fff5f0;
              font-weight: 700;
            }
            .secondary {
              background: var(--accent-soft);
              color: #743624;
              font-weight: 700;
            }
            .status-grid, .results {
              display: grid;
              grid-template-columns: repeat(3, minmax(0, 1fr));
              gap: 16px;
            }
            .card {
              padding: 18px;
              min-height: 440px;
              display: flex;
              flex-direction: column;
            }
            .card h2 {
              margin: 0;
              font-size: 1.2rem;
            }
            .note {
              margin: 8px 0 12px;
              color: var(--muted);
              font-size: 0.94rem;
            }
            .badge {
              display: inline-block;
              padding: 4px 10px;
              border-radius: 999px;
              font-size: 0.85rem;
              font-weight: 700;
            }
            .ready { background: #d7f5dc; color: #14532d; }
            .pending { background: #f7d9d6; color: #7f1d1d; }
            .poem {
              flex: 1;
              margin-top: 14px;
              padding: 16px;
              border-radius: 16px;
              border: 1px dashed var(--edge);
              background: rgba(255,255,255,0.55);
              font-family: "IBM Plex Mono", "Fira Code", monospace;
              white-space: pre-wrap;
              line-height: 1.55;
              overflow: auto;
            }
            .meta {
              margin-top: 14px;
              color: var(--muted);
              font-size: 0.88rem;
            }
            .footer {
              margin-top: 18px;
              color: var(--muted);
              font-size: 0.95rem;
            }
            @media (max-width: 1180px) {
              .controls { grid-template-columns: 1fr 1fr; }
              .status-grid, .results { grid-template-columns: 1fr; }
            }
          </style>
        </head>
        <body>
          <main class="shell">
            <h1>Poetry LM Compare</h1>
            <p class="lead">
              Сравнение трех текущих веток генерации по одной первой строке. Веб-приложение живет на
              `ebekkuev.runningdog.org`, а инференс остается на отдельной GPU-машине и вызывается через proxy.
            </p>

            <section class="panel">
              <div class="controls">
                <div>
                  <label for="prompt">Первая строка</label>
                  <textarea id="prompt" placeholder="Жизнь просит, требует словами,"></textarea>
                </div>
                <div>
                  <label for="temperature">Temperature</label>
                  <input id="temperature" type="range" min="0.1" max="1.5" step="0.05" value="0.9">
                  <div class="range-meta"><span>0.1</span><strong id="temperature-value">0.90</strong><span>1.5</span></div>
                </div>
                <div>
                  <label for="top-k">Top-k</label>
                  <input id="top-k" type="range" min="1" max="100" step="1" value="50">
                  <div class="range-meta"><span>1</span><strong id="top-k-value">50</strong><span>100</span></div>
                </div>
                <button id="generate" class="primary">Сгенерировать</button>
                <button id="refresh" class="secondary">Обновить</button>
              </div>
            </section>

            <section class="status-grid" id="status-grid"></section>
            <section class="results" id="results"></section>

            <p class="footer">
              Параметры, скрытые в этой версии: `max_new_tokens=160`. Если понадобится воспроизводимость,
              следующим шагом стоит добавить `seed`.
            </p>
          </main>

          <script>
            const resultsEl = document.getElementById("results");
            const statusEl = document.getElementById("status-grid");
            const promptEl = document.getElementById("prompt");
            const tempEl = document.getElementById("temperature");
            const topkEl = document.getElementById("top-k");
            const tempValueEl = document.getElementById("temperature-value");
            const topkValueEl = document.getElementById("top-k-value");

            tempEl.addEventListener("input", () => {
              tempValueEl.textContent = Number(tempEl.value).toFixed(2);
            });
            topkEl.addEventListener("input", () => {
              topkValueEl.textContent = topkEl.value;
            });

            function renderStatus(models) {
              statusEl.innerHTML = models.map((model) => `
                <article class="card" style="min-height: 190px">
                  <div class="badge ${model.ready ? "ready" : "pending"}">${model.ready ? "ready" : "pending"}</div>
                  <h2 style="margin-top:12px">${model.title}</h2>
                  <p class="note">${model.note}</p>
                  <div class="meta"><div><strong>checkpoint</strong><br><code>${model.checkpoint}</code></div></div>
                </article>
              `).join("");
              resultsEl.innerHTML = models.map((model) => `
                <article class="card" data-model-key="${model.key}">
                  <div class="badge ${model.ready ? "ready" : "pending"}">${model.ready ? "ready" : "pending"}</div>
                  <h2 style="margin-top:12px">${model.title}</h2>
                  <p class="note">${model.note}</p>
                  <pre class="poem">${model.ready ? "Готово к генерации." : "Чекпойнт еще не готов."}</pre>
                </article>
              `).join("");
            }

            async function refreshStatus() {
              const resp = await fetch("/api/health");
              if (!resp.ok) {
                throw new Error(`health failed: ${resp.status}`);
              }
              const payload = await resp.json();
              renderStatus(payload.models);
              return payload.models;
            }

            async function generate() {
              const prompt = promptEl.value.trim();
              if (!prompt) {
                alert("Нужна первая строка.");
                return;
              }
              const cards = [...document.querySelectorAll("[data-model-key]")];
              for (const card of cards) {
                card.querySelector(".poem").textContent = "Генерация...";
              }
              for (const card of cards) {
                const modelKey = card.dataset.modelKey;
                try {
                  const resp = await fetch("/api/generate", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({
                      model_key: modelKey,
                      prompt,
                      temperature: Number(tempEl.value),
                      top_k: Number(topkEl.value)
                    })
                  });
                  const payload = await resp.json();
                  if (!resp.ok) {
                    card.querySelector(".poem").textContent = payload.detail || `HTTP ${resp.status}`;
                    continue;
                  }
                  card.querySelector(".poem").textContent = payload.output;
                } catch (err) {
                  card.querySelector(".poem").textContent = String(err);
                }
              }
            }

            document.getElementById("generate").addEventListener("click", generate);
            document.getElementById("refresh").addEventListener("click", refreshStatus);
            promptEl.addEventListener("keydown", (event) => {
              if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                generate();
              }
            });

            refreshStatus().catch((err) => {
              statusEl.innerHTML = `<article class="card"><h2>Health error</h2><pre class="poem">${String(err)}</pre></article>`;
            });
          </script>
        </body>
        </html>
        """
    )


def build_app(inference_api_base_url: str) -> FastAPI:
    app = FastAPI(title="Poetry LM Web App", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return build_index_html()

    @app.get("/api/health")
    async def api_health():
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(f"{inference_api_base_url}/health")
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        return JSONResponse(status_code=resp.status_code, content=resp.json())

    @app.post("/api/generate")
    async def api_generate(request: GenerateRequest):
        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                resp = await client.post(
                    f"{inference_api_base_url}/generate",
                    json=request.model_dump(),
                )
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        return JSONResponse(status_code=resp.status_code, content=resp.json())

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--inference-api-base-url", default=DEFAULT_INFERENCE_API_BASE_URL)
    args = parser.parse_args()

    app = build_app(args.inference_api_base_url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
