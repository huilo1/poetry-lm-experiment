# Poetry LM

Экспериментальный проект по обучению с нуля русскоязычной поэтической causal LM, которая получает первую строку и продолжает стихотворение в заданном формате.

Текущие исследовательские ветки:

- `AABB CCDD` baseline на 8 строк;
- `ABAB ABAB` comparative branch;
- staged training: сначала полные стихи, потом строгое `AABB CCDD` дообучение.

Основные компоненты:

- подготовка корпуса из `IlyaGusev/stihi_ru`;
- stress-aware проверка рифмы;
- собственный `SentencePiece`;
- небольшой `decoder-only Transformer`;
- локальный инференс и compare UI;
- deployable web app для домена и отдельный inference API для GPU-хоста.

## Структура

- `src/poetry_lm/`: модель, токенизация, рифма, инференс
- `scripts/`: датасеты, обучение, генерация, HTTP API, web UI
- `configs/`: train-конфиги
- `deploy/`: `systemd`, `nginx`, env templates и install scripts
- `DEPLOY_AGENT.md`: подробные инструкции отдельному агенту по развертыванию
- `HANDOFF.md`: журнал экспериментов, метрики и промежуточные выводы

## Локальный быстрый старт

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
python scripts/download_stihi.py
python scripts/build_dataset.py --input data/raw/stihi_ru.jsonl.zst
python scripts/train_tokenizer.py
python scripts/prepare_tokens.py
python scripts/train.py --config configs/tiny_cpu.json
python scripts/generate.py --checkpoint artifacts/checkpoints/smoke_cpu/best.pt --tokenizer-model artifacts/tokenizer/poetry.model --prompt "Я помню чудное мгновенье"
```

## Compare UI

Локальная морда для сравнения веток:

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/web_ui.py --host 127.0.0.1 --port 7860
```

## Deploy схема

Публичный домен живет на web-хосте, а инференс остается на отдельной GPU-машине:

- `scripts/web_app.py`: веб-приложение для `ebekkuev.runningdog.org`
- `scripts/inference_api.py`: HTTP inference API на GPU-хосте
- `deploy/systemd/poetry-inference-tunnel.service`: SSH tunnel web-host -> GPU-host
- `deploy/nginx/ebekkuev.runningdog.org.conf`: reverse proxy для Nginx

Полная инструкция по разворачиванию: [DEPLOY_AGENT.md](/home/angel/projects/Poetry/DEPLOY_AGENT.md)
