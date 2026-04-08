from __future__ import annotations

import argparse
import json
import runpy
import sys
import tempfile
from pathlib import Path

from poetry_lm.gigachat_sft import prepare_gigachat_local_model_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["base_model"] = prepare_gigachat_local_model_dir(cfg["base_model"])

    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
        tmp_config = fh.name

    sys.argv = [
        str(Path(__file__)),
        "--config",
        tmp_config,
    ]
    if args.resume:
        sys.argv.extend(["--resume", args.resume])

    runpy.run_path(str(Path(__file__).with_name("train_qwen_sft.py")), run_name="__main__")


if __name__ == "__main__":
    main()
