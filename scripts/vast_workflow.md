# Vast Workflow

1. Установить CLI:

```bash
uv tool install vastai
vastai --help
```

2. Найти GPU:

```bash
vastai search offers 'reliability > 0.98 num_gpus=1 gpu_ram>=24 dph<0.6 inet_up>100 inet_down>100'
```

3. Создать инстанс:

```bash
vastai create instance <offer_id> --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime --disk 80
```

4. Залить проект, поставить зависимости, обучить:

```bash
rsync -av . <instance>:/workspace/Poetry/
ssh <instance> 'cd /workspace/Poetry && pip install -e . && python scripts/train.py --config configs/vast_small.json'
```

5. Скачать артефакты и удалить инстанс:

```bash
rsync -av <instance>:/workspace/Poetry/artifacts ./artifacts
vastai destroy instance <instance_id>
```
