# Poetry LM Handoff

This file preserves the current project state so work can continue after context loss.

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
  - `Stage1 -> Stage2 current`

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
