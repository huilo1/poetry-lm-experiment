from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from poetry_lm.inference import generate_text, load_bundle, resolve_device
from poetry_lm.model_registry import model_map, model_specs


MAX_NEW_TOKENS = 160


class GenerateRequest(BaseModel):
    model_key: str
    prompt: str = Field(min_length=1, max_length=400)
    temperature: float = Field(default=0.9, ge=0.1, le=1.5)
    top_k: int = Field(default=50, ge=1, le=100)


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


def build_app(device: str) -> FastAPI:
    resolved_device = "cpu" if device == "auto" and training_is_running() else resolve_device(device)
    models = model_map()
    app = FastAPI(title="Poetry LM Inference API", version="0.1.0")

    @app.get("/health")
    def health():
        items = []
        for spec in model_specs():
            items.append(
                {
                    "key": spec.key,
                    "title": spec.title,
                    "note": spec.note,
                    "checkpoint": str(spec.checkpoint),
                    "tokenizer": str(spec.tokenizer),
                    "ready": spec.checkpoint.exists() and spec.tokenizer.exists(),
                }
            )
        return {
            "ok": True,
            "device": resolved_device,
            "max_new_tokens": MAX_NEW_TOKENS,
            "training_running": training_is_running(),
            "models": items,
        }

    @app.post("/generate")
    def generate(request: GenerateRequest):
        spec = models.get(request.model_key)
        if spec is None:
            raise HTTPException(status_code=404, detail="unknown model key")
        if not spec.checkpoint.exists():
            raise HTTPException(status_code=409, detail="checkpoint not ready")
        if not spec.tokenizer.exists():
            raise HTTPException(status_code=500, detail="tokenizer missing")

        bundle = load_bundle(spec.checkpoint, spec.tokenizer, device=resolved_device)
        try:
            output = generate_text(
                bundle=bundle,
                prompt=request.prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=request.temperature,
                top_k=request.top_k,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "model_key": spec.key,
            "title": spec.title,
            "device": resolved_device,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "top_k": request.top_k,
            "output": output,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    app = build_app(args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
