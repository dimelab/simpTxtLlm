# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python CLI research tool for analyzing discourse in large text corpora using Ollama and open-source LLMs. The concrete task it has been built around is **binary classification of paragraphs for the presence of a discourse position** (e.g. anti-establishment stance) plus a free-text position label, but the framing is user-supplied via prompt files so it is domain-agnostic.

Pipeline: documents → semantic segmentation → LLM evaluation → human review → few-shot/full fine-tuning, with embedding-similarity search and label normalization as auxiliary tools.

`implementation_plan.md` describes the original spec; the code has since grown beyond it (see "Commands" — `score`, `search-similar`, `normalize-positions`, `stats` are not in the plan). Treat the code as the source of truth.

## Key Design Constraints

- **Polars only** — no pandas anywhere.
- **No async** — sequential processing throughout (one `ollama.chat` call at a time).
- **No database / no server** — flat files only (JSON, parquet, CSV); CLI via Typer.
- **Ollama** — `ollama` Python library for chat/inference; `subprocess` for `ollama create` (the library does not expose Modelfile creation).
- **Sentence embeddings** — `sentence-transformers`, default model `all-MiniLM-L6-v2` (override with `--embedding-model`; multilingual e5 models recommended for Danish, see README).
- Research-tool simplicity — avoid over-engineering.

## Running

Run everything through the top-level entrypoint (it just imports `discourse_tool.cli.app`):

```bash
python cli.py <command> [options]
```

Requires Ollama installed and `ollama serve` running. Install deps with `pip install -r requirements.txt`. There is no test suite, linter config, or build step — this is a single-author research tool. A `.venv/` (Python 3.9) is checked into the tree.

## Commands

| Command | Module fn | Purpose |
|---|---|---|
| `segment` | `segment.segment_files` / `segment_csv` | Split docs into semantic paragraphs → JSON |
| `evaluate` | `evaluate.evaluate_segments` | Run each paragraph through an Ollama model → parquet+csv |
| `intermediary-evaluate` | `intermediary.intermediary_evaluate` | Second-pass evaluator re-assesses flagged positives → CLEAR/BORDERLINE/ABSENT + filter + drift/anchor diagnostics |
| `review` | `finetune.review_evaluations` | Interactive human confirmation of `binary_flag` → parquet |
| `stats` | inline in `cli.py` | Print counts / flag distribution / position breakdown |
| `score` | `finetune.score_model` | Train/test split (by article), build few-shot model, report accuracy/precision/recall/F1 |
| `search-similar` | `similarity.search_similar` | Rank target segments by cosine similarity to per-position centroid embeddings |
| `normalize-positions` | `similarity.normalize_positions` | Collapse noisy position labels to a canonical list via `difflib` fuzzy matching |
| `finetune` | `finetune.finetune` | `few-shot` (enriched-prompt model) or `full` (JSONL + axolotl config export) |

CLI option details and examples live in `README.md` — keep it in sync when changing command signatures.

## Architecture & Cross-Cutting Conventions

These are the things that span multiple files and are easy to break:

- **The `|||` output protocol.** Models are expected to return `binary_flag ||| position ||| reason`. `evaluate.py` parses it (3-part split; all three columns null and `raw_evaluation` preserved if parsing fails). `finetune.py` and `score_model` *re-emit* this same format when building few-shot examples and JSONL. Any change to the delimiter or field order must be made in all three places. The system prompt / user template you write must instruct the model to produce this format.

- **`binary_flag` / `human_flag` are strings `"0"`/`"1"`, not ints.** All comparisons (`== "1"`) and filters assume this. The "positive" class is `"1"`.

- **Custom Ollama models are content-addressed by prompt hash.** `evaluate.py` names the model `{base_model}-discourse-{md5(system_prompt)[:8]}`; few-shot/score variants use `-discourse-fewshot-` / `-discourse-score-`. Re-running with the same prompt reuses the same model name (idempotent `ollama create`). A Modelfile is written into the output dir as a side effect.

- **30-character minimum segment filter** is applied consistently in `evaluate`, `review`, and `search-similar` to skip noise. Keep it consistent if you touch one.

- **Resumability is built in, keyed differently per command.** `evaluate` skips whole articles already present (by `source_file`) and appends, saving every 1000 paragraphs; `--restart` discards. `review` skips already-reviewed `(source_file, paragraph_index)` pairs and saves after *every* segment (Ctrl-C safe). Preserve these semantics — they're the main UX guarantee.

- **Embedding cache (`similarity.py` only).** Embeddings are cached next to the source file as `{stem}_embeddings.npz` + `{stem}_embeddings_meta.json`, invalidated by source mtime, model name, or text-list mismatch. `--restart` forces recompute.

- **`intermediary-evaluate` is a deliberate exception to the `|||` protocol.** It sends *batches* of cases per `ollama.chat` call and parses a **JSON array** back (with a `[`…`]` substring fallback). Because batches are large, these calls **must** pass `options={"num_ctx": context_window}` — Ollama defaults to `num_ctx=2048` and would silently truncate a batch otherwise. The label shown to the evaluator is `binary_flag` + `position` only; the analyst's `reason` is intentionally excluded to avoid anchoring. Outputs (evaluated/filtered/calibration/anchors) land in `data/intermediary/`. Resumes like `evaluate` (skips done `(source_file, paragraph_index)`, saves after every batch); calibration/anchor reports describe the most recent run only.

- **`config.py`** is a single `Config` dataclass holding default paths (`data/{segments,evaluations,training,similarity}`), `default_model="mistral"`, `similarity_threshold=0.3`, `embedding_model`, and the `modelfile_template`. Directory properties are derived; callers default missing args to `Config()` values rather than threading config through.

## Data Flow & Schemas

```
docs/CSV → segment  → data/segments/<stem>.json   {"<id>": ["para", ...]}
         → evaluate → data/evaluations/<stem>.{parquet,csv}
                       cols: source_file, paragraph_index, text,
                             binary_flag, position, reason, raw_evaluation
         → intermediary-evaluate → data/intermediary/<stem>_evaluated.{parquet,csv}
                       + tier, confidence, evidence, strongest_feature, weakest_feature
                       (+ _filtered, _calibration, _anchors)
         → review   → data/training/human_evaluations.parquet
                       + human_flag, accepted
         → finetune → few-shot Ollama model  OR  training_data.jsonl + axolotl_config.yml
         → search-similar → data/similarity/<stem>_similarity.{parquet,csv}
                       cols: source_file, paragraph_index, text, position, similarity
```

`segment` accepts a single file, a directory (PDF/DOCX/TXT), or a CSV (requires `--id-column` and `--text-column`). CSV mode writes one combined JSON; file mode writes one JSON per input file.
