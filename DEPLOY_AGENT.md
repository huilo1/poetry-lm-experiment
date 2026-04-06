# Deploy Agent Instructions

Ниже инструкции для отдельного агента, который будет разворачивать публичную веб-морду на `http://ebekkuev.runningdog.org` и оставлять инференс на отдельной GPU-машине.

## Цель

Нужно развернуть такую схему:

- публичный домен `ebekkuev.runningdog.org` живет на web-хосте;
- web-хост поднимает `scripts/web_app.py`;
- web-хост не хранит модели и не делает локальный инференс;
- web-хост проксирует запросы на GPU-хост через локальный SSH tunnel;
- GPU-хост поднимает `scripts/inference_api.py`;
- inference API использует уже существующие модели в `artifacts/checkpoints/*` и токенайзеры в `artifacts/tokenizer_*/*`.

## Что уже есть в репозитории

- `scripts/inference_api.py`: HTTP API для генерации на GPU-машине
- `scripts/web_app.py`: публичная веб-морда для домена
- `deploy/systemd/*.service`: systemd unit files
- `deploy/nginx/ebekkuev.runningdog.org.conf`: конфиг Nginx
- `deploy/env/*.example`: шаблоны env-файлов
- `deploy/scripts/install_gpu_inference_service.sh`
- `deploy/scripts/install_web_stack.sh`

## Важное ограничение

Модели и датасеты **не** лежат в GitHub. Они остаются вне репозитория. Поэтому:

- на GPU-хосте сервис нужно стартовать из существующего каталога, где уже лежат `artifacts/checkpoints` и `artifacts/tokenizer_*`;
- web-хосту нужен только код и Python-окружение.

## Часть 1. GPU-хост

### 1. Проверь рабочий каталог

Нужно использовать каталог, где уже есть:

- `artifacts/checkpoints/host_5060_8line_20m/best.pt`
- `artifacts/checkpoints/host_5060_8line_abab_20m/best.pt`
- при готовности staged ветки:
  - `artifacts/checkpoints/host_5060_aabb_qf2_stage2_from_fullpoem_20m/best.pt`
- `artifacts/tokenizer_aabb8/poetry.model`
- `artifacts/tokenizer_abab8/poetry.model`

Если это по-прежнему `/home/angel/projects/Poetry`, используй именно его как `PROJECT_ROOT`.

### 2. Обнови код из GitHub

Если это existing clone:

```bash
cd /home/angel/projects/Poetry
git pull
```

Если это не clone, а просто рабочая директория, сначала привяжи ее к репозиторию или сделай отдельный clone и перенеси только код. Не перемещай модели без необходимости.

### 3. Подготовь окружение

```bash
cd /home/angel/projects/Poetry
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

### 4. Создай env-файл

```bash
sudo mkdir -p /etc/poetry-lm
sudo cp deploy/env/gpu-inference-api.env.example /etc/poetry-lm/gpu-inference-api.env
sudoedit /etc/poetry-lm/gpu-inference-api.env
```

Проверь значения:

- `PROJECT_ROOT=/home/angel/projects/Poetry`
- `INFERENCE_API_HOST=127.0.0.1`
- `INFERENCE_API_PORT=8001`
- `INFERENCE_DEVICE=cuda`

### 5. Установи и запусти systemd сервис

```bash
cd /home/angel/projects/Poetry
bash deploy/scripts/install_gpu_inference_service.sh
```

### 6. Проверь API локально

```bash
curl http://127.0.0.1:8001/health
```

Ожидается JSON со списком моделей и статусами `ready/pending`.

## Часть 2. Web-хост

### 1. Клонируй репозиторий

Пример:

```bash
cd /opt
git clone <GITHUB_REPO_URL> poetry-lm
cd poetry-lm
```

### 2. Подготовь окружение

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

### 3. Подготовь SSH-доступ к GPU-хосту

Нужен ключ, который позволяет web-хосту выполнить:

```bash
ssh -p 2222 angel@<GPU_HOST>
```

Если в вашей схеме GPU-хост достигается как `localhost:2222`, настрой именно это. Главное, чтобы tunnel-сервис мог достучаться до GPU-хоста без интерактивного ввода пароля.

### 4. Создай env-файлы

```bash
sudo mkdir -p /etc/poetry-lm
sudo cp deploy/env/web-app.env.example /etc/poetry-lm/web-app.env
sudo cp deploy/env/inference-tunnel.env.example /etc/poetry-lm/inference-tunnel.env
sudoedit /etc/poetry-lm/web-app.env
sudoedit /etc/poetry-lm/inference-tunnel.env
```

Заполни их так:

`/etc/poetry-lm/web-app.env`

- `PROJECT_ROOT=/opt/poetry-lm`
- `WEBAPP_HOST=127.0.0.1`
- `WEBAPP_PORT=8080`
- `INFERENCE_API_BASE_URL=http://127.0.0.1:8001`

`/etc/poetry-lm/inference-tunnel.env`

- `SSH_USER=angel`
- `SSH_HOST=<GPU_HOST>`
- `SSH_PORT=2222`
- `SSH_KEY_PATH=/home/<deploy-user>/.ssh/id_ed25519`
- `TUNNEL_LOCAL_PORT=8001`
- `GPU_API_PORT=8001`

### 5. Установи Nginx, если он еще не установлен

```bash
sudo apt-get update
sudo apt-get install -y nginx
```

### 6. Установи web stack

```bash
cd /opt/poetry-lm
bash deploy/scripts/install_web_stack.sh
```

### 7. Проверь локально на web-хосте

```bash
curl http://127.0.0.1:8080/
curl http://127.0.0.1:8080/api/health
curl http://127.0.0.1:8001/health
```

Где:

- `127.0.0.1:8080` это локальный `web_app.py`
- `127.0.0.1:8001` это локальный конец SSH tunnel

### 8. Проверка Nginx

```bash
sudo nginx -t
sudo systemctl status --no-pager nginx
sudo systemctl status --no-pager poetry-inference-tunnel.service
sudo systemctl status --no-pager poetry-web-app.service
```

### 9. DNS и внешний доступ

Убедись, что:

- `ebekkuev.runningdog.org` указывает на web-хост;
- порт `80` доступен извне;
- если потом включается TLS, добавь certbot поверх уже созданного `nginx` vhost.

## Smoke checklist

После деплоя нужно проверить:

1. `curl http://127.0.0.1:8001/health` на web-хосте возвращает health inference API через tunnel.
2. `curl http://127.0.0.1:8080/api/health` на web-хосте возвращает тот же JSON через web app proxy.
3. В браузере `http://ebekkuev.runningdog.org` открывается страница с тремя карточками моделей.
4. Генерация работает хотя бы для `AABB CCDD baseline`.
5. Если staged чекпойнт еще не готов, интерфейс корректно показывает `pending`, а не падает.

## Не делать

- Не копировать в GitHub `data/raw`, `data/processed*`, `.venv`, `artifacts/checkpoints`, `*.bin`, крупные `jsonl.gz`.
- Не удалять существующие чекпойнты на GPU-хосте.
- Не запускать web app на GPU-хосте вместо inference API.
- Не открывать inference API наружу напрямую, если достаточно SSH tunnel + web app proxy.
