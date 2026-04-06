# Poetry LM Handoff

This file preserves the current project state so work can continue after context loss.

## New Active Direction

The next serious experiment is no longer another scratch variant, but a base-model comparison branch:
- target base model: `Qwen/Qwen3-8B-Base`
- method: completion-style QLoRA / SFT
- task stays the same: input `1` line, generate `8` lines total, scheme `AABB CCDD`

Implemented for this branch:
- dataset converter: `scripts/build_qwen_sft_dataset.py`
- training script: `scripts/train_qwen_sft.py`
- generation script: `scripts/generate_qwen.py`
- evaluation script: `scripts/evaluate_qwen_8line.py`
- helper module: `src/poetry_lm/qwen_sft.py`
- config: `configs/vast_qwen3_8b_aabb_qf2_qlora.json`

Training format used for Qwen:
- plain-text structured tags, no custom tokenizer
- prompt prefix:
  - `[TASK]`
  - `[FORMAT]`
  - `[SCHEME]`
  - `[L1]`
  - `[GEN]`
  - `[L2]`
- model is trained to complete lines `2-8` only

Reason for this direction:
- scratch-only stack appears close to its useful ceiling
- strongest remaining hypothesis is that better language quality requires a strong Russian-capable base model

### Qwen branch status

Current Qwen dataset:
- `data/qwen_aabb_qf2_sft`
- built from `data/processed_aabb8_qf2`
- stats:
  - train `67180`
  - val `5232`
  - test `2172`

Qwen training format:
- prompt prefix:
  - `[TASK] Продолжи русское стихотворение.`
  - `[FORMAT] Ровно 8 строк.`
  - `[SCHEME] AABB CCDD`
  - `[L1] ...`
  - `[GEN]`
  - `[L2]`
- supervised target is lines `2-8`

Smoke validation completed successfully on Vast:
- config: `configs/smoke_qwen3_0_6b_aabb_qf2_lora.json`
- base model: `Qwen/Qwen3-0.6B-Base`
- result: training loop completed end-to-end
- best eval loss: about `2.7669`
- train runtime: about `63s`

Important infra fix discovered during smoke:
- `bitsandbytes` / `triton` required a system C compiler inside the remote container
- fix on remote host: install `build-essential` before QLoRA training

Main run now in progress on Vast:
- config: `configs/vast_qwen3_8b_aabb_qf2_qlora.json`
- base model: `Qwen/Qwen3-8B-Base`
- runtime: Vast instance `34242647`
- host: `ssh9.vast.ai:12646`
- remote train process: `python scripts/train_qwen_sft.py --config configs/vast_qwen3_8b_aabb_qf2_qlora.json`
- remote artifacts directory: `/workspace/Poetry/artifacts/checkpoints/vast_qwen3_8b_aabb_qf2_qlora`
- current state when this handoff was updated: weights fetch started (`Fetching 5 files...`)

### Qwen throughput finding

The first `8B` QLoRA config turned out to be too conservative and underutilized the GPU.

Measured on Vast `RTX 4090 49GB`:
- old mode: QLoRA (`load_in_4bit=true`) with checkpointing
- observed VRAM: only about `15 GB`
- observed wall time: about `22 sec / optimizer step`
- this implied roughly `50+ hours` train time for the planned 2-epoch run

Benchmark then confirmed a much faster regime:
- config: `configs/benchmark_qwen3_8b_aabb_lora_bf16.json`
- mode: plain LoRA in `bf16`, no `4bit`, no gradient checkpointing
- batch: `8`
- observed VRAM: about `44.8 GB`
- observed GPU utilization: `100%`
- observed step time: about `0.7 sec / step`

Decision:
- switch the main Qwen run away from QLoRA
- use full-GPU LoRA bf16 instead
- new main config: `configs/vast_qwen3_8b_aabb_qf2_lora_bf16.json`

### Autonomous watcher

To avoid manual polling, a local watcher was added:
- script: `scripts/watch_qwen_vast_run.sh`
- responsibility:
  - poll remote train/eval status
  - sync remote checkpoint directory to local `artifacts/downloaded/vast_qwen3_8b_aabb_qf2_lora_bf16`
  - destroy Vast instance `34242647` after artifacts are synced

Runtime bookkeeping:
- pid file: `artifacts/logs/watch_qwen_vast_run.pid`
- status log: `artifacts/logs/watch_qwen_vast_run.log`
- stdout/stderr: `artifacts/logs/watch_qwen_vast_run.stdout.log`

## Agreed Next Step After Qwen

Once the current `Qwen3-8B-Base` branch is fully evaluated, the next comparison branch is:
- model family: `GigaChat`
- variant: `base`, not `instruct`
- target candidate: `ai-sage/GigaChat3-10B-A1.8B-base`
- training protocol: same `LoRA` setup style as the final successful Qwen run
- purpose: clean `Qwen base vs GigaChat base` comparison under the same poetry task and the same evaluation protocol

Important constraints for that next step:
- do **not** switch to instruct for the primary comparison run
- keep the same task definition and as much of the same dataset / metric pipeline as possible
- inference should first be tested locally on the controlled `RTX 5060 Ti 8GB`
- if local inference is not viable, move inference to a rented GPU instance

## Final Qwen Result

Branch:
- `Qwen/Qwen3-8B-Base`
- `LoRA bf16`
- config: `configs/vast_qwen3_8b_aabb_qf2_lora_bf16.json`

Training summary:
- train runtime: about `10360.7s` (`2h 53m`)
- best eval loss: `1.9239`
- train loss: `1.9981`
- local summary copy: `artifacts/downloaded/vast_qwen3_8b_aabb_qf2_lora_bf16/summary.json`

Task evaluation on `300` prompts:
- `exact_8_lines_rate = 0.98`
- `second_line_rhyme_rate = 0.0233`
- `aabb_ccdd_rate = 0.0`
- local eval copy: `artifacts/downloaded/vast_qwen3_8b_aabb_qf2_lora_bf16/eval8.json`

Interpretation:
- the base model easily learned the fixed 8-line output length
- but it did not internalize the required rhyme scheme under the current SFT format
- compared to the scratch `AABB CCDD` baseline, it is dramatically worse on rhyme and scheme despite much better language fluency
- this is an important negative result: for the strict rhyme-constrained task, the scratch poetry-only model outperformed the strong general-purpose base model

## Goal

Russian-only poetry continuation model trained from scratch.

Current agreed next target:
- input: 1 line
- output: 8 lines total
- rhyme scheme: `AABB CCDD`
- no reranker for now
- add stress-aware rhyme detection

## What Was Built

Current training/inference stack:
- tokenizer: SentencePiece unigram, vocab `16000`
- model: decoder-only Transformer from scratch
- main model size: about `39.8M` params
- training code: `scripts/train.py`
- generation code: `scripts/generate.py`
- evaluation code: `scripts/evaluate.py`

Main source files:
- `scripts/build_dataset.py`
- `scripts/build_rhyme_focus_dataset.py`
- `scripts/prepare_tokens.py`
- `scripts/train.py`
- `scripts/generate.py`
- `scripts/evaluate.py`
- `src/poetry_lm/rhyme.py`
- `src/poetry_lm/tokenizer.py`
- `src/poetry_lm/model.py`
- `src/poetry_lm/training.py`

## Dataset History

### 1. Baseline dataset: `data/processed`

Built from `IlyaGusev/stihi_ru`, filtered to poems with detectable quatrain rhyme.

Stats from `data/processed/stats.json`:
- `rows_total`: `5151050`
- `rows_kept`: `1219940`
- `train_size`: `1098242`
- `val_size`: `85377`
- `test_size`: `36321`
- `scheme_ABAB`: `852200`
- `scheme_AABB`: `315152`
- `scheme_ABBA`: `52588`

Token stats from `data/processed/meta.json`:
- `train`: `219388601`
- `val`: `16929938`
- `test`: `7270126`

### 2. Rhyme-focused dataset: `data/processed_rhyme_focus`

Purpose:
- keep only `AABB` poems
- oversample the first rhyming pair via extra 2-line examples

Builder:
- `scripts/build_rhyme_focus_dataset.py`

Stats from `data/processed_rhyme_focus/stats.json`:
- `train_full_poems`: `283907`
- `train_pair_examples`: `567814`
- `train_rows_written`: `851721`
- `val_full_poems`: `21935`
- `val_pair_examples`: `43870`
- `test_full_poems`: `9310`
- `test_pair_examples`: `18620`

Token stats from `data/processed_rhyme_focus/meta.json`:
- `train`: `70793921`
- `val`: `5446883`
- `test`: `2293023`

Important side effect:
- this dataset strongly improved second-line rhyme
- but it also biased generation toward very short outputs, often just 2 lines

## Training Runs

### 1. Baseline run: `vast_main`

Config:
- `configs/vast_main.json`

Artifacts:
- `artifacts/downloaded/vast_main/best.pt`
- `artifacts/downloaded/vast_main/final.pt`
- `artifacts/downloaded/vast_main/log.jsonl`
- `artifacts/downloaded/vast_main/train.log`
- `artifacts/downloaded/vast_main/best.eval.json`
- `artifacts/downloaded/vast_main/best.generate.json`

Best observed validation:
- best `val_loss`: about `3.2320`

Evaluation on original test set:
- `second_line_rhyme_rate = 0.12`

Inference note:
- model often produced longer continuations
- rhyme quality was weak

### 2. Rhyme-focused retrain: `vast_rhyme_focus`

Config:
- `configs/vast_rhyme_focus.json`

Artifacts:
- `artifacts/downloaded/vast_rhyme_focus/best.pt`
- `artifacts/downloaded/vast_rhyme_focus/final.pt`
- `artifacts/downloaded/vast_rhyme_focus/log.jsonl`
- `artifacts/downloaded/vast_rhyme_focus/train.log`
- `artifacts/downloaded/vast_rhyme_focus/best.eval.json`
- `artifacts/downloaded/vast_rhyme_focus/best.generate.json`

Final validation:
- `val_loss = 2.9589931964874268` at `iter 12000`

Evaluation on the same original test set:
- `second_line_rhyme_rate = 0.55`

Practical effect:
- rhyme improved dramatically
- generation became too short and often stopped after 2 lines

Example from `best.generate.json`:
- prompt: `Я помню чудное мгновенье`
- output: `Я помню чудное мгновенье / В любви нашей творенье.`

## Infrastructure Notes

Local preprocessing happens on this host.

Training workflow used:
- preprocess locally
- rent Vast instance via CLI
- train remotely
- download artifacts
- destroy instance

Inference server:
- `ssh angel@localhost -p 2222`

Inference bundle already deployed there:
- `/home/angel/projects/Poetry/artifacts/checkpoints/vast_rhyme_focus/best.pt`

Local server environment:
- Python `3.12`
- GPU `RTX 5060 Ti 8GB`

Observed training VRAM for the `~39.8M` model:
- about `9.2 GB` on `RTX 4090`
- about `9.35 GB` on earlier `RTX 5090` run

Conclusion:
- current training config is too large for local `8 GB` VRAM without shrinking batch/context/model
- inference is fine locally

## Known Issues

1. Current rhyme detector is weak.
- `src/poetry_lm/rhyme.py` uses simple word-tail heuristics
- no stress dictionary yet
- likely introduces noisy labels

2. Rhyme-focused dataset is too aligned to the first pair only.
- improves line `1-2`
- harms longer structured generation

3. Generation is unconstrained.
- no explicit length target
- no explicit line-position tokens
- no explicit rhyme-class tokens

## Current Agreed Direction

Next iteration target:
- exactly `8` lines
- rhyme scheme `AABB CCDD`
- input is still `1` line
- no reranker yet
- add stress-aware rhyme logic

Recommended changes for the next implementation:

1. Replace current rhyme labeling with stress-aware detection.
- add dictionary-based stress support
- compute rhyme tails from the stressed vowel

2. Build a new task-specific dataset.
- keep only 8-line windows with valid `AABB CCDD`
- allow windows extracted from longer poems, not only full poems
- remove the 2-line oversampling used in `processed_rhyme_focus`

3. Change the training text format.
- add explicit structural tokens like:
  - `<PROMPT>`
  - `<GEN>`
  - `<LEN_8>`
  - `<SCHEME_AABB_CCDD>`
  - line position tokens like `<L1>` ... `<L8>`

4. Evaluate against the new task, not only second-line rhyme.
- exact 8-line rate
- exact `AABB CCDD` rate
- per-pair rhyme rate for `1-2`, `3-4`, `5-6`, `7-8`
- early EOS rate
- repeated-line rate

## Commands That Were Useful

Local inference on the controlled server:

```bash
ssh angel@localhost -p 2222 '
cd /home/angel/projects/Poetry
. .venv/bin/activate
PYTHONPATH=src python scripts/generate.py \
  --checkpoint artifacts/checkpoints/vast_rhyme_focus/best.pt \
  --tokenizer-model artifacts/tokenizer/poetry.model \
  --device cuda \
  --prompt "И день и ночь и снегопад" \
  --attempts 32 \
  --max-new-tokens 64
'
```

## Summary

Best completed result so far:
- training from scratch works
- rhyme-focused retraining improved second-line rhyme from `0.12` to `0.55`
- this came at the cost of collapsing many generations to short 2-line outputs

The next step is not decoder reranking, but a more faithful supervised objective:
- 1 input line
- 8 output lines
- `AABB CCDD`
- stress-aware rhyme labels

## Completed 8-Line Iteration

This iteration is now completed.

### Dataset

New task-specific dataset:
- `data/processed_aabb8`

Task definition used:
- input: `1` line
- output: `8` lines
- scheme: `AABB CCDD`

Dataset stats from `data/processed_aabb8/stats.json`:
- `train_windows_kept`: `80253`
- `val_windows_kept`: `6301`
- `test_windows_kept`: `2637`

Token stats from `data/processed_aabb8/meta.json`:
- `train`: `8993090`
- `val`: `706603`
- `test`: `295979`
- `vocab_size`: `16000`

Quality-filtered version:
- `data/processed_aabb8_qf2`

Built by:
- `scripts/filter_aabb8_quality.py`
- `src/poetry_lm/quality.py`

Purpose:
- remove structurally weak 8-line windows
- remove duplicate-line / same-last-word rhyme hacks
- penalize strong cross-rhyme collisions
- remove fragmentary endings and obvious lexical noise

Final qf2 stats from `data/processed_aabb8_qf2/stats.json`:
- `train_rows_kept`: `67180`
- `val_rows_kept`: `5232`
- `test_rows_kept`: `2172`
- `train_keep_rate`: `0.8371`
- `val_keep_rate`: `0.8303`
- `test_keep_rate`: `0.8237`

Main rejection causes:
- `fragment_end`
- `same_last_word_in_pair`
- `duplicate_line`
- `pair_meter_mismatch`
- score penalties from `cross_rhyme_collision`, `high_word_repetition`, `low_lexical_diversity`

Artifacts produced:
- `data/processed_aabb8_qf2/train.jsonl.gz`
- `data/processed_aabb8_qf2/val.jsonl.gz`
- `data/processed_aabb8_qf2/test.jsonl.gz`
- `data/processed_aabb8_qf2/stats.json`
- `data/processed_aabb8_qf2/reject_examples.json`
- `data/processed_aabb8_qf2/*.kept.preview.txt`
- `data/processed_aabb8_qf2/*.rejected.preview.txt`

### Code added for this iteration

- stress-aware accents:
  - `src/poetry_lm/stress.py`
- stress-aware rhyme detection:
  - `src/poetry_lm/rhyme.py`
- structured 8-line tokenizer format:
  - `src/poetry_lm/tokenizer.py`
- 8-line dataset builder:
  - `scripts/build_aabb8_dataset.py`
- 8-line evaluator:
  - `scripts/evaluate_8line.py`
- quality filter v2:
  - `src/poetry_lm/quality.py`
  - `scripts/filter_aabb8_quality.py`
- training configs:
  - `configs/vast_8line_20m.json`
  - `configs/vast_8line_24m.json`
  - `configs/host_5060_8line_20m.json`

### Final training run

Run name:
- `host_5060_8line_20m`

Why local instead of Vast:
- Vast SSH/container startup was unreliable in this session
- local `RTX 5060 Ti 8GB` was sufficient for the `20M` model

Model:
- decoder-only Transformer
- about `20.4M` params
- config: `8` layers, `384` embd, `6` heads

Training artifacts:
- `artifacts/checkpoints/host_5060_8line_20m/best.pt`
- `artifacts/checkpoints/host_5060_8line_20m/final.pt`
- `artifacts/checkpoints/host_5060_8line_20m/log.jsonl`
- `artifacts/checkpoints/host_5060_8line_20m/train.log`
- `artifacts/checkpoints/host_5060_8line_20m/best.eval8.json`
- `artifacts/checkpoints/host_5060_8line_20m/best.generate.json`

Best checkpoint:
- `iter_num = 10500`
- `best_val_loss = 2.778297185897827`

Training time:
- about `1372` sec to `12000` iters

Observed local GPU memory on clean run:
- probe peak about `3554 MiB` at `batch_size=20`, `gradient_accumulation_steps=4`, `float16`

### Evaluation for the 8-line task

From `artifacts/checkpoints/host_5060_8line_20m/best.eval8.json` on `300` test samples:
- `exact_8_lines_rate = 0.9966666666666667`
- `aabb_ccdd_rate = 0.41333333333333333`
- `second_line_rhyme_rate = 0.76`

Interpretation:
- length control is now essentially solved
- second-line rhyme is stronger than the previous rhyme-focused run
- full `AABB CCDD` structure is achieved in about `41%` of samples without reranking
- text quality is still noisy and sometimes degenerates lexically

### Important bug fixed after training

The model was generating correct structure, but evaluation initially reported zeros because decoding was wrong.

Root cause:
- remote host had two package copies:
  - `/home/angel/projects/Poetry/poetry_lm/`
  - `/home/angel/projects/Poetry/src/poetry_lm/`
- imports on the remote host resolved to the root copy
- old `Tokenizer.decode()` collapsed line boundaries

Fix applied:
- synced the updated package into both remote locations
- `Tokenizer.decode()` now treats both `<NL>` and `<L2>...<L8>` as line boundaries

### Example generations

Example:
- prompt: `Любовь опасна и сильна,`
- output:
  - `Любовь опасна и сильна,`
  - `Любовь всегда своя весна.`
  - `⁇ онара, она без прикрас,`
  - `Как много хочется чудес,`
  - `Коль я полон зла и потерь.`
  - `Она лишает только дверь,`
  - `Кто душу к жизни придаёт,`
  - `Она в ней свою обойдёт.`

Example:
- prompt: `Жизнь просит, требует словами,`
- output:
  - `Жизнь просит, требует словами,`
  - `Не ополчёшься своими, делами.`
  - `Поддержит тело на лучшее`
  - `И повернется, - не ведемся мы в тире.`
  - `У каждого есть повторение,`
  - `И выделяет наслаждение.`
  - `И кто нам, не пожнёшь, не отпустишь,`
  - `И в то время, когда всё уснёшь.`

### Most likely next step

The next highest-leverage improvement is:
- retrain the same `20M` model on `data/processed_aabb8_qf2`
- compare against the current `host_5060_8line_20m` baseline on:
  - `exact_8_lines_rate`
  - `aabb_ccdd_rate`
  - `second_line_rhyme_rate`
- only after that decide whether decode-time reranking is still necessary
  - `src/poetry_lm/tokenizer.py`
- new dataset builder for 8-line task:
  - `scripts/build_aabb8_dataset.py`
- new evaluator for 8-line task:
  - `scripts/evaluate_8line.py`
- new train config:
- `configs/vast_8line_aabb.json`

### Important design decision

Original strict interpretation of `AABB AABB` as one global rhyme class `A` across lines `1,2,5,6` and one global class `B` across lines `3,4,7,8` made the dataset almost collapse.

So the current implementation uses the practical interpretation:
- first quatrain must be `AABB`
- second quatrain must be `AABB`
- no requirement that the first quatrain `A` rhyme matches the second quatrain `A`

Machine scheme label used in code:
- `AABB_CCDD`

### Current prompt/training format

Structured control tokens are used:
- `<PROMPT>`
- `<GEN>`
- `<LEN_8>`
- `<SCHEME_AABB_CCDD>`
- `<L1>` ... `<L8>`

Current training text shape:

```text
<PROMPT> <LEN_8> <SCHEME_AABB_CCDD> <L1> first line <NL> <GEN> <L2> second line <NL> ... <L8> eighth line
```

### Stress logic

Key practical insight:
- `ruaccent-predictor` gives bad outputs for some bare words like `мгновенье`
- but works correctly for `last_word + "."`

So current stress-aware rhyme path accents only the last word, not the whole line.

Example:
- `снегопад` -> `снегопа'д`
- `белизною` -> `белизно'ю`
- `град` stays `град`, which is acceptable because it is monosyllabic

### Smoke build result

Smoke dataset:
- built from `data/processed`
- only first 8 lines of each poem
- `max-poems=1000` per split
- output dir: `data/processed_aabb8_smoke`

Stats from `data/processed_aabb8_smoke/stats.json`:
- train kept: `82 / 1000`
- val kept: `86 / 1000`
- test kept: `80 / 1000`

More detailed stats:
- `train_windows_total`: `937`
- `train_windows_kept`: `82`
- `train_windows_rejected_heuristic`: `839`
- `train_windows_rejected_stress`: `16`

This shows:
- the task is not too sparse anymore
- data density is around `8%` on the broad processed corpus
- the implementation is viable

### Runtime updates

The first stress-aware builder version was too slow because it called `ruaccent` effectively per-window.

Two important fixes were added:
- `scripts/generate.py` no longer reranks or retries by rhyme. Generation is now a single honest sample.
- `scripts/build_aabb8_dataset.py` now does chunked stress validation:
  - heuristic filter first
  - collect candidate 8-line windows in batches
  - accent last words for the whole batch in one `ruaccent` GPU call
  - validate `AABB_CCDD` from precomputed stressed tails

Measured timings:
- local CPU stress benchmark for `200` candidate windows: about `65.6s` (`~3.05 rows/s`)
- `localhost:2222` GPU stress benchmark for `200` candidate windows: about `14.3s` (`~13.98 rows/s`)
- remote end-to-end builder benchmark with batched stress:
  - `--max-poems 5000`
  - full `train+val+test` completed in `1:06.07`

This changed the full-build estimate from many hours to roughly about `1 hour`.

### Practical dataset expectation

From the `5000`-row remote benchmark:
- train kept: `500`
- val kept: `465`
- test kept: `481`

That implies the final strict `AABB_CCDD` corpus will likely be much smaller than the old rhyme-focused corpus, on the order of roughly `~85k` train windows if the keep-rate stays similar.

This matters for model sizing:
- the old `~39.8M` config may now be too large for the effective token budget
- likely candidates for the next retrain are:
  - `20.4M` params: `8 layers, 384 embd`
  - `23.9M` params: `10 layers, 384 embd`
- previous larger configs for reference:
  - `33.5M`: `8 layers, 512 embd`
  - `39.8M`: `10 layers, 512 embd`

### Practical runtime conclusion

Using all sliding 8-line windows across the full corpus is too expensive locally.

Current practical build strategy:
- use only the first 8 lines of each poem by default
- keep sliding windows as an optional mode via `--all-windows`

### Remaining work for this iteration

1. Build the full `data/processed_aabb8` dataset.
2. Confirm final dataset size and token count.
3. Train a new tokenizer for the structured format, likely into `artifacts/tokenizer_aabb8/`.
4. Pack tokens for `data/processed_aabb8`.
5. Choose final model size based on actual token count, likely `20M-24M`.
6. Train with `configs/vast_8line_aabb.json` or a smaller derived config.
7. Run `scripts/evaluate_8line.py`.
8. Deploy the new checkpoint to `angel@localhost:2222`.

## ABAB branch

The project was then generalized to test whether `ABAB ABAB` is a better 8-line target than `AABB CCDD`.

### Fast scheme frequency checks

Quick heuristic counts on a sample of `45000` poems from `data/processed` (`571898` 8-line windows total):
- `ABAB + ABAB`: `31192`
- `AABB + AABB`: `21209`
- `ABBA + ABBA`: `1340`

This means:
- `ABAB ABAB` appears about `1.47x` more often than `AABB CCDD`
- `ABBA CDDC` is much rarer and is not a promising main target

### Comparable smoke build

Stress-aware smoke builds with `--all-windows` and `--max-poems 1000` per split showed:

- `ABAB` kept:
  - train `462`
  - val `435`
  - test `559`
- `AABB` kept:
  - train `424`
  - val `422`
  - test `350`

Interpretation:
- `ABAB` has a lower keep-rate
- but it still yields more valid windows in absolute terms because the source pattern is more common

### Code changes for ABAB

Generalization work already landed:
- `src/poetry_lm/rhyme.py`
  - added `detect_eight_line_abab_abab_from_tails`
  - added `detect_eight_line_abab_abab`
- `src/poetry_lm/tokenizer.py`
  - added `STRUCTURED_MODE_ABAB = "structured_8line_abab_abab"`
  - added `STRUCTURED_SCHEME_ABAB = "ABAB_ABAB"`
  - added control token `<SCHEME_ABAB_ABAB>`
  - `structured_window_to_training_text(...)` is now scheme-aware
  - `format_prompt`, `encode_prompt`, and `decode` now support both schemes
- `src/poetry_lm/quality.py`
  - generalized to accept a target scheme via `score_window(..., scheme=...)`
  - rhyme pair indices for `ABAB` are `(0,2),(1,3),(4,6),(5,7)`
- `scripts/filter_aabb8_quality.py`
  - now accepts `--default-scheme`
- `scripts/evaluate_8line.py`
  - now supports both prompt modes and reports `abab_abab_rate` for the ABAB branch
- new builder:
  - `scripts/build_abab8_dataset.py`
- new train config:
  - `configs/host_5060_8line_abab_20m.json`

### Remote ABAB pipeline

The full ABAB branch is running on the inference server at `angel@localhost:2222`.

Remote paths:
- PID file: `/home/angel/projects/Poetry/artifacts/logs/abab_pipeline.pid`
- Log file: `/home/angel/projects/Poetry/artifacts/logs/abab_pipeline.log`
- Shell wrapper: `/home/angel/projects/Poetry/artifacts/logs/run_abab_pipeline.sh`

Pipeline steps:
1. `python scripts/build_abab8_dataset.py --input-dir data/processed --out-dir data/processed_abab8 --all-windows --stress-batch-size 256`
2. `python scripts/filter_aabb8_quality.py --input-dir data/processed_abab8 --out-dir data/processed_abab8_qf2 --stress-batch-size 256 --default-scheme ABAB_ABAB`
3. `python scripts/train_tokenizer.py --input data/processed_abab8_qf2/train.jsonl.gz --tmp-text data/processed_abab8_qf2/tokenizer_input.txt --model-prefix artifacts/tokenizer_abab8/poetry --vocab-size 16000`
4. `python scripts/prepare_tokens.py --dataset-dir data/processed_abab8_qf2 --tokenizer-model artifacts/tokenizer_abab8/poetry.model`
5. `python scripts/train.py --config configs/host_5060_8line_abab_20m.json`

Current observed state during this handoff:
- the remote process is alive
- it is still in `build_train`
- latest observed progress in the log was about `24923` processed train poems after about `7` minutes

Given the size of `data/processed/train.jsonl.gz`, this full run is expected to take materially longer than the earlier focused branches.

### Strongest next change without an external base model

If ABAB helps but language quality is still noisy, the next strongest change should likely be **staged training on the same poetry corpus**:

1. train the same architecture on full poems from the same Russian poetry corpus
2. then continue training from that checkpoint on the strict 8-line structured task

Why:
- it should improve syntax and lexical stability
- it does not rely on a ready-made base model
- it does not require a broader general Russian corpus
- it keeps the experiment faithful to "train from scratch on poetry"

### ABAB full run results

The full remote `ABAB ABAB` pipeline on `angel@localhost:2222` completed successfully.

Preprocessing timeline:
- pipeline shell script created: `2026-04-05 22:59:53 +0300`
- `data/processed_abab8/train.jsonl.gz`: `2026-04-06 04:19:12 +0300`
- `data/processed_abab8/val.jsonl.gz`: `2026-04-06 04:44:15 +0300`
- `data/processed_abab8/test.jsonl.gz`: `2026-04-06 04:55:03 +0300`
- `data/processed_abab8_qf2/train.jsonl.gz`: `2026-04-06 09:04:56 +0300`
- `data/processed_abab8_qf2/val.jsonl.gz`: `2026-04-06 09:24:26 +0300`
- `data/processed_abab8_qf2/test.jsonl.gz`: `2026-04-06 09:32:48 +0300`
- tokenizer built around `2026-04-06 09:33:50 +0300`
- packed token files built around `2026-04-06 09:34:12-09:34:15 +0300`
- training started around `2026-04-06 09:34-09:35 +0300`

Artifacts:
- checkpoint dir: `artifacts/checkpoints/host_5060_8line_abab_20m/`
- best checkpoint: `artifacts/checkpoints/host_5060_8line_abab_20m/best.pt`
- final checkpoint: `artifacts/checkpoints/host_5060_8line_abab_20m/final.pt`
- eval file: `artifacts/checkpoints/host_5060_8line_abab_20m/best.eval8.json`
- tokenizer: `artifacts/tokenizer_abab8/poetry.model`

Training results:
- model size is the same local `20M` class used for comparison
- `max_iters = 12000`
- final logged iteration: `12000`
- best validation loss: `2.2915244102478027`

Evaluation on `300` test examples from `data/processed_abab8_qf2/test.jsonl.gz`:
- `exact_8_lines_rate = 0.9966666666666667`
- `abab_abab_rate = 0.30333333333333334`
- `second_line_rhyme_rate = 0.04`

Comparison against the current `AABB CCDD` baseline (`host_5060_8line_20m`):
- both models keep `8` lines almost perfectly (`0.9967`)
- `AABB` full-scheme rate was `0.4133`
- `ABAB` full-scheme rate is `0.3033`
- `AABB` second-line rhyme rate was `0.76`
- `ABAB` second-line rhyme rate is only `0.04`

Interpretation:
- `ABAB` is more frequent in the corpus, but it is much harder for this model/prompting setup
- because the rhyme target skips one line, the model is currently much weaker at preserving the first rhyme class across a gap
- this makes `ABAB` a worse immediate target than `AABB CCDD`, despite the larger dataset

Practical conclusion after this branch:
- keep `AABB CCDD` as the stronger current baseline
- preserve `ABAB` datasets and checkpoints as important negative-but-informative experimental artifacts for the paper
- the next strongest idea remains staged training on full poems first, then strict 8-line task fine-tuning on top

## Staged training branch

After the `ABAB` result, the next main experiment was changed to:

1. stage 1 pretraining on full poems from the same poetry corpus
2. stage 2 fine-tuning on the strict `AABB CCDD` 8-line task

The goal is to improve lexical and syntactic stability without using:
- a ready-made Russian base model
- a broad non-poetry Russian corpus

### Code changes for staged training

Two pipeline changes were added:

- `scripts/train.py`
  - new `--init-from` flag
  - unlike `--resume`, it loads only model weights and resets optimizer/schedule
- `scripts/prepare_tokens.py`
  - now supports separate `--input-dir` and `--output-dir`
  - this allows one tokenizer to be reused across multiple datasets without duplicating raw JSONL files

New configs:
- `configs/host_5060_fullpoem_20m_stage1.json`
- `configs/host_5060_aabb_qf2_stage2_from_fullpoem_20m.json`

New orchestration script:
- `scripts/run_stage1_stage2_pipeline.sh`

### Important tokenizer decision

The first staged attempt tried to train a brand new full-poem tokenizer, but this turned out to be unnecessary and expensive.

Verification showed:
- `artifacts/tokenizer/poetry.model` does **not** contain structured tokens properly (`<PROMPT>`, `<GEN>`, `<L1>` etc. map to `<unk>`)
- `artifacts/tokenizer_aabb8/poetry.model` **does** contain the structured tokens correctly

Therefore the staged branch was changed to **reuse**:
- `artifacts/tokenizer_aabb8/poetry.model`

This avoids a very slow extra SentencePiece step and keeps vocabulary compatible between stage 1 and stage 2.

### Remote staged run status

Host:
- `angel@localhost:2222`

Continuation log:
- `/home/angel/projects/Poetry/artifacts/logs/stage1_stage2_cont.log`

Continuation PID:
- `/home/angel/projects/Poetry/artifacts/logs/stage1_stage2_cont.pid`

Stage 1 output dir:
- `/home/angel/projects/Poetry/artifacts/checkpoints/host_5060_fullpoem_20m_stage1`

Stage 2 output dir:
- `/home/angel/projects/Poetry/artifacts/checkpoints/host_5060_aabb_qf2_stage2_from_fullpoem_20m`

Prepared datasets:
- full-poem tokenized dataset:
  - `data/processed_fullpoem_stage1tok/train.bin`
  - `data/processed_fullpoem_stage1tok/val.bin`
  - `data/processed_fullpoem_stage1tok/test.bin`
  - `data/processed_fullpoem_stage1tok/meta.json`
- structured stage 2 tokenized dataset:
  - `data/processed_aabb8_qf2_stage1tok/*`

Current observed status at handoff time:
- stage 2 token packing already completed
- stage 1 training is running
- first logged stage 1 metrics:
  - iter `0`: train `9.7556`, val `9.7542`
  - iter `500`: train `4.8362`, val `4.8342`
  - iter `1000`: train `4.4855`, val `4.4832`

### Experimental caveat

This staged branch is not a perfectly controlled comparison against the current `AABB` baseline yet, because:
- the current best `AABB` baseline was trained from scratch on `processed_aabb8`
- the staged branch uses:
  - `processed_aabb8_qf2`
  - `tokenizer_aabb8`
  - initialization from stage 1

If staged training looks promising, a clean control run should later be added:
- same `processed_aabb8_qf2`
- same `tokenizer_aabb8`
- same `20M` architecture
- but **without** stage 1 initialization

## Web UI

A comparison web UI was added to make manual model checking easier while experiments are still running.

Files:
- `scripts/web_ui.py`
- `src/poetry_lm/inference.py`

Related updates:
- `scripts/generate.py` now reuses the shared inference helper
- `pyproject.toml` now includes `gradio`

Current UI behavior:
- one input line from the user
- controls:
  - `temperature`
  - `top_k`
- outputs:
  - `AABB CCDD baseline`
  - `ABAB ABAB branch`
  - `AABB planner-guided`

UI / API registry update:
- the failed staged branch was removed from the active comparison UI
- the new planner-guided branch was added instead
- this required changes in:
  - `src/poetry_lm/model_registry.py`
  - `scripts/inference_api.py`
  - `scripts/web_ui.py`
  - `scripts/web_app.py`
- model registry now supports multi-checkpoint specs:
  - generator checkpoint
  - optional planner checkpoint
- the public web app status cards now show both checkpoints for planner-guided models

Important runtime decision:
- if `--device auto` is used and a `python scripts/train.py` process is detected, the UI falls back to `cpu`
- this avoids fighting the active training run for GPU memory

Remote deployment:
- files were synced to `angel@localhost:2222`
- `gradio` was installed into the remote `.venv`
- `scripts/web_ui.py --help` was verified successfully on the remote host

Suggested run command on the remote host:
- `cd /home/angel/projects/Poetry`
- `. .venv/bin/activate`
- `PYTHONPATH=src python scripts/web_ui.py --host 127.0.0.1 --port 7860`

If access from the local machine is needed, tunnel it with SSH:
- `ssh -L 7860:127.0.0.1:7860 angel@localhost -p 2222`

## GitHub repo

The project was converted into a git repository and pushed to GitHub:

- repo URL: `https://github.com/huilo1/poetry-lm-experiment`

Important repository decisions:
- large data and artifacts are excluded via `.gitignore`
- the repo contains:
  - source code
  - training / preprocessing scripts
  - deployment assets
  - experiment notes in `HANDOFF.md`
  - deployment instructions in `DEPLOY_AGENT.md`
- the repo does **not** contain:
  - raw datasets
  - processed corpora
  - checkpoints
  - tokenized `.bin` files
  - virtual environments

Deploy-related files now present in the repo:
- `scripts/inference_api.py`
- `scripts/web_app.py`
- `deploy/env/*.example`
- `deploy/systemd/*.service`
- `deploy/nginx/ebekkuev.runningdog.org.conf`
- `deploy/scripts/install_gpu_inference_service.sh`
- `deploy/scripts/install_web_stack.sh`
- `DEPLOY_AGENT.md`

## Final staged result

The staged branch finished fully and produced a useful negative result.

Stage 1:
- checkpoint dir: `artifacts/checkpoints/host_5060_fullpoem_20m_stage1`
- best / final validation loss observed at the end: about `3.4233`

Stage 2:
- checkpoint dir: `artifacts/checkpoints/host_5060_aabb_qf2_stage2_from_fullpoem_20m`
- best validation loss: `2.6673` at `iter 5000`
- final validation loss: `2.6787`
- eval file: `artifacts/checkpoints/host_5060_aabb_qf2_stage2_from_fullpoem_20m/best.eval8.json`

Stage 2 eval on 300 test prompts:
- `exact_8_lines_rate = 0.9867`
- `second_line_rhyme_rate = 0.58`
- `aabb_ccdd_rate = 0.1567`

Comparison against the current AABB baseline:
- baseline:
  - `exact_8_lines_rate = 0.9967`
  - `second_line_rhyme_rate = 0.76`
  - `aabb_ccdd_rate = 0.4133`
- staged:
  - `0.9867 / 0.58 / 0.1567`

Conclusion:
- staged pretraining on full poems improved `val_loss`
- but it **hurt** the actual constrained 8-line rhymed generation task
- this is an important negative result for the paper

## Ending planner branch

A new experimental branch was implemented to split the task into:
1. planning endings for lines `2 / 4 / 6 / 8`
2. generating the poem conditioned on those planned endings

Core idea:
- planner predicts target end words from the first line only
- generator receives those planned endings before `<GEN>`
- this is a real architectural change, not a decode-time reranker

New tokenizer modes and control tokens:
- file: `src/poetry_lm/tokenizer.py`
- new modes:
  - `planner_8line_aabb_ccdd`
  - `structured_8line_aabb_ccdd_plan`
- new tokens:
  - `<PLAN>`
  - `<E2>`
  - `<E4>`
  - `<E6>`
  - `<E8>`

New helper modules:
- `src/poetry_lm/planning.py`
- `src/poetry_lm/inference.py` now supports:
  - `generate_plan(...)`
  - `generate_text(..., plan_endings=...)`
  - `generate_text_with_planner(...)`

New scripts:
- dataset builder:
  - `scripts/build_aabb_plan_datasets.py`
- planner inference:
  - `scripts/generate_with_plan.py`
- planner eval:
  - `scripts/evaluate_plan_endings.py`
- end-to-end planner + generator eval:
  - `scripts/evaluate_planned_8line.py`

New configs:
- full runs:
  - `configs/host_5060_aabb_end_planner_12m.json`
  - `configs/host_5060_aabb_with_plan_20m.json`
- smoke runs:
  - `configs/smoke_aabb_end_planner.json`
  - `configs/smoke_aabb_with_plan.json`

New intermediate datasets:
- planner dataset target dir:
  - `data/processed_aabb8_plan`
- generator-with-plan target dir:
  - `data/processed_aabb8_planned`

Smoke verification already completed:
- built tiny datasets:
  - `data/processed_aabb8_plan_smoke`
  - `data/processed_aabb8_planned_smoke`
- trained smoke tokenizer:
  - `artifacts/tokenizer_aabb_plan_smoke/poetry.model`
- packed smoke `.bin` files for both datasets
- trained smoke planner checkpoint:
  - `artifacts/checkpoints/smoke_aabb_end_planner/best.pt`
- trained smoke generator checkpoint:
  - `artifacts/checkpoints/smoke_aabb_with_plan/best.pt`
- verified end-to-end command:
  - `PYTHONPATH=src python scripts/generate_with_plan.py ...`

Important note from smoke:
- the planner can emit incomplete plans early in training
- the inference path was adjusted so the generator can still run with partially filled `<E2>/<E4>/<E6>/<E8>`
- quality of smoke outputs is meaningless; the point is that the full pipeline is now working

Recommended full pipeline commands:

1. Build planner and generator datasets from the current best strict corpus:
- `PYTHONPATH=src python scripts/build_aabb_plan_datasets.py --input-dir data/processed_aabb8_qf2 --planner-out-dir data/processed_aabb8_plan --generator-out-dir data/processed_aabb8_planned`

2. Train a shared tokenizer on both branches:
- `PYTHONPATH=src python scripts/train_tokenizer.py --input data/processed_aabb8_plan/train.jsonl.gz data/processed_aabb8_planned/train.jsonl.gz --tmp-text data/processed_aabb8_plan/tokenizer_input.txt --model-prefix artifacts/tokenizer_aabb_plan/poetry --vocab-size 16000`

3. Pack planner tokens:
- `PYTHONPATH=src python scripts/prepare_tokens.py --dataset-dir data/processed_aabb8_plan --tokenizer-model artifacts/tokenizer_aabb_plan/poetry.model`

4. Pack generator-with-plan tokens:
- `PYTHONPATH=src python scripts/prepare_tokens.py --dataset-dir data/processed_aabb8_planned --tokenizer-model artifacts/tokenizer_aabb_plan/poetry.model`

5. Train planner:
- `PYTHONPATH=src python scripts/train.py --config configs/host_5060_aabb_end_planner_12m.json`

6. Evaluate planner endings:
- `PYTHONPATH=src python scripts/evaluate_plan_endings.py --checkpoint artifacts/checkpoints/host_5060_aabb_end_planner_12m/best.pt --tokenizer-model artifacts/tokenizer_aabb_plan/poetry.model --test-file data/processed_aabb8_plan/test.jsonl.gz --device cuda --limit 300`

7. Train generator with plan:
- `PYTHONPATH=src python scripts/train.py --config configs/host_5060_aabb_with_plan_20m.json`

8. Evaluate planner-guided full generation:
- `PYTHONPATH=src python scripts/evaluate_planned_8line.py --planner-checkpoint artifacts/checkpoints/host_5060_aabb_end_planner_12m/best.pt --generator-checkpoint artifacts/checkpoints/host_5060_aabb_with_plan_20m/best.pt --tokenizer-model artifacts/tokenizer_aabb_plan/poetry.model --test-file data/processed_aabb8_planned/test.jsonl.gz --device cuda --limit 300`

Full run result on the GPU host:

Planner:
- checkpoint dir: `artifacts/checkpoints/host_5060_aabb_end_planner_12m`
- best validation loss: `2.3660` at `iter 3000`
- eval:
  - `ending_exact_match_rate = 0.0158`
  - `ending_rhyme_tail_match_rate = 0.1525`

Generator with plan:
- checkpoint dir: `artifacts/checkpoints/host_5060_aabb_with_plan_20m`
- best validation loss: `2.5538` at `iter 12000`
- eval:
  - `exact_8_lines_rate = 0.9967`
  - `second_line_rhyme_rate = 0.63`
  - `aabb_ccdd_rate = 0.3167`

Comparison versus the current strict AABB baseline:
- baseline:
  - `exact_8_lines_rate = 0.9967`
  - `second_line_rhyme_rate = 0.76`
  - `aabb_ccdd_rate = 0.4133`
- planner-guided:
  - `0.9967 / 0.63 / 0.3167`

Conclusion:
- planner-guided is clearly better than the failed staged branch
- but it is still worse than the current AABB baseline
- it keeps length just as well, but does not improve rhyme quality enough
