# Implementation Plan — Assisted Intermediary Evaluation Step

## 1. Purpose

A second-pass LLM "methodological evaluator" that independently re-assesses the
analyst model's initial coding. For each flagged paragraph it judges whether the
assigned label (`binary_flag` + `position`) is **supported by the source text**,
given the original analytical framework — without seeing the analyst's `reason`
(to prevent anchoring on the analyst's argument).

Output: every evaluated case tagged `CLEAR` / `BORDERLINE` / `ABSENT` with a
confidence score and evidence, plus a filtered subset of cases that pass a
threshold, plus drift/consistency diagnostics.

Inspired by `add_ons/batch_evaluator_v2.md`; the section below records where that
draft must change to fit this codebase.

## 2. Evaluation of the add_ons draft against the current codebase

| # | add_ons draft | Reality in this repo | Decision |
|---|---|---|---|
| 1 | Reads the label from `row["model_evaluation"]` | No such column. Schema is `binary_flag, position, reason, raw_evaluation` | Reconstruct the label string from `binary_flag` + `position`; **exclude `reason`** (anti-anchoring) |
| 2 | `ollama.chat(...)` with no options | Ollama defaults to `num_ctx=2048`; 30–50-case batches would be silently truncated | **Pass `options={"num_ctx": context_window}`** on every call. Real bug in the draft. |
| 3 | `pl.read_parquet(evaluations_path)` only | `evaluate` writes both `.parquet` and `.csv`; user's file is `.csv` | Accept both — branch on suffix (`.csv` → `read_csv`, `.parquet` → `read_parquet`) |
| 4 | No incremental save; everything held in memory, saved once at end | Core repo convention: `evaluate` and `review` both resume + save periodically. 627k rows makes an all-or-nothing run fragile | **Add resumable batches**: persist results after each batch; on re-run skip already-evaluated `case_id`s. `--restart` to force fresh. |
| 5 | `case_id = "{source_file}__p{paragraph_index}"` echoed by the model | `source_file` is a long URL; LLMs echo long ids unreliably in JSON | Assign **short integer `case_id`s** per run, keep a `case_id → (source_file, paragraph_index)` lookup, join back on the integer |
| 6 | `_case_id_to_index()` referenced | Never defined in the draft | Implement via the lookup dict above |
| 7 | No minimum-length filter | Repo consistently skips segments `< 30` chars | Apply the same `len(text) >= 30` filter |
| 8 | No system prompt wired into `ollama.chat` (only a user message); evaluator system prompt file exists separately | Repo bakes system prompts into a custom Ollama model via Modelfile (`create_modelfile`/`create_ollama_model`), content-addressed by `md5(prompt)[:8]` | Reuse that path: bake `evaluator_system_prompt_v2.txt` into a custom model `{model}-evaluator-{hash}`; send only the per-batch user message |
| 9 | Evaluates all rows | User decision: **only `binary_flag == "1"`**, with an optional position allow-list | Filter to positives; if `--position/-p` given (repeatable), keep only those positions |
| 10 | `random.shuffle` with fixed seed | Repo uses `random` module (no seed) but a seed is fine and aids reproducibility | Keep `--seed` (default 42) |

Net: the draft's **architecture is sound** (batch + comparative calibration +
anchors + drift report), but it must be adapted to the real schema, the
Modelfile/`num_ctx` conventions, resumability, robust case ids, the 30-char
filter, and the positives-only + position-allow-list scope.

## 3. Scope (confirmed with user)

- **Cases evaluated:** only `binary_flag == "1"`, `len(text) >= 30`.
- **Optional position allow-list:** `--position/-p` (repeatable). When supplied,
  discard positives whose `position` is not in the list. (Run
  `normalize-positions` first to collapse noisy labels.)
- **Label shown to evaluator:** `binary_flag` + `position` only (no `reason`).
- **Diagnostics:** full v2 machinery — overlap/anchor consistency + per-batch
  calibration/drift report.

## 4. New module: `discourse_tool/intermediary.py`

Functions (mirrors the draft, with the fixes above):

- `load_evaluations(path) -> pl.DataFrame` — branch on `.csv`/`.parquet`.
- `select_cases(df, positions, min_len=30) -> pl.DataFrame` — filter to
  `binary_flag=="1"`, length, and optional position allow-list.
- `build_label(row) -> str` — `f"binary_flag={row['binary_flag']}; position={row['position']}"`.
- `build_batch_items(df_batch, id_map) -> str` — XML `<case id="N">…` blocks
  using short integer ids; label from `build_label` (no `reason`).
- `build_evaluator_user_message(batch_items, analyst_system, analyst_user, evaluator_user_template) -> str`
  — fills `{system_prompt}`, `{user_template}`, `{batch_items}`.
- `estimate_batch_size(df, context_window) -> int` — token heuristic from the
  draft, but using `build_label` instead of `model_evaluation`; clamp `[10, 50]`.
- `create_batches(df, batch_size, overlap, shuffle, seed) -> list[dict]` — as in
  draft (`indices`, `anchor_indices`, `new_indices`).
- `parse_evaluator_response(content) -> list[dict]` — `json.loads` with
  `[`…`]` substring fallback (from draft).
- `evaluate_cases(...)` — main loop:
  - Bake evaluator system prompt into custom model (`create_modelfile` /
    `create_ollama_model`, name `{base_model}-evaluator-{hash}`).
  - For each batch: build user message, `ollama.chat(model, messages,
    options={"num_ctx": context_window})`, parse JSON, map `case_id` → row.
  - Record anchor classifications across batches; store first result per case.
  - **Persist after each batch** to `{stem}_evaluated.parquet`/`.csv`; skip
    `case_id`s already present unless `--restart`.
- `calibration_report(results, batches, id_map) -> pl.DataFrame` — per-batch
  CLEAR/BORDERLINE/ABSENT %, avg confidence, deviation + `drift_flag` (>0.20).
- `anchor_consistency(anchor_classifications) -> pl.DataFrame` — per anchor
  case: list of classifications across batches + `consistent` bool.
- `intermediary_evaluate(...)` — orchestrator: load → select → batch → evaluate
  → join evaluator columns back onto the selected df → write outputs → filter →
  print diagnostics.

### Output files (in `data/intermediary/`, new `Config.intermediary_dir`)

| File | Contents |
|---|---|
| `{stem}_evaluated.parquet` / `.csv` | all selected cases + `tier`, `confidence`, `evidence`, `strongest_feature`, `weakest_feature` |
| `{stem}_filtered.parquet` / `.csv` | cases passing `--threshold` (`clear`→CLEAR only; `borderline`→CLEAR+BORDERLINE) and `--min-confidence` |
| `{stem}_calibration.parquet` | per-batch distribution + drift flags |
| `{stem}_anchors.parquet` | anchor cross-batch consistency |

## 5. Prompt files

Move the v2 prompts into the repo's prompt convention (out of `add_ons/`):
- `prompts/evaluator_system.txt` ← `evaluator_system_prompt_v2.txt` (becomes the
  Modelfile SYSTEM). **Update its output spec to use `tier` and include
  `case_id`**, matching parsing.
- `prompts/evaluator_user_template.txt` ← `evaluator_user_template_v2.txt`.
- The analyst system prompt + user template are passed in via CLI flags (same
  files used for the original `evaluate` run).

## 6. `config.py` change

Add `intermediary_dir` property (`data/intermediary`) and include it in
`ensure_dirs()`. Add a `context_window` default (e.g. `32000`) and
`evaluator_overlap`/`evaluator_seed` defaults if we want them centralized.

## 7. CLI command (`discourse_tool/cli.py`)

New command `intermediary-evaluate` (alias the draft's `filter`; chosen name
avoids clashing with the verb "filter" and reads as the pipeline step):

```bash
python cli.py intermediary-evaluate \
  --evaluations data/evaluations/extract_articles_denmark_all.csv \
  --analyst-system-prompt prompts/system.txt \
  --analyst-user-template prompts/template.txt \
  --evaluator-system-prompt prompts/evaluator_system.txt \
  --evaluator-user-template prompts/evaluator_user_template.txt \
  --model mistral \
  -p "anti-establishment" -p "economic critique" \   # optional allow-list
  --batch-size auto --context-window 32000 \
  --overlap 5 --shuffle --seed 42 \
  --threshold clear --min-confidence 4 \
  --output data/intermediary/ \
  --restart
```

Flags: `--evaluations/-e`, `--analyst-system-prompt`, `--analyst-user-template`,
`--evaluator-system-prompt`, `--evaluator-user-template`, `--model/-m`,
`--position/-p` (repeatable, optional), `--batch-size` (int or `auto`),
`--context-window`, `--overlap`, `--shuffle/--no-shuffle`, `--seed`,
`--threshold` (`clear|borderline`), `--min-confidence`, `--output/-o`,
`--restart`. Body imports `intermediary` lazily (matches existing commands).

## 8. Docs

- README: add a "Intermediary evaluation" section between Evaluate and Review
  documenting the command, the CLEAR/BORDERLINE/ABSENT scheme, and the outputs.
- CLAUDE.md: add the new command to the command table and the new module;
  note the JSON-batch protocol is a deliberate exception to the `|||` protocol,
  and that `num_ctx` must be set for batched calls.

## 9. Build order

1. `config.py` — `intermediary_dir` + defaults.
2. Move/adjust prompt files into `prompts/`.
3. `intermediary.py` — data loading, selection, batching, prompt assembly.
4. `intermediary.py` — evaluate loop with Modelfile + `num_ctx` + incremental save.
5. `intermediary.py` — calibration + anchor reports + orchestrator + filtering.
6. `cli.py` — wire the `intermediary-evaluate` command.
7. README + CLAUDE.md.

## 10. Open risks / notes

- **Runtime:** even positives-only, this is many batched LLM calls; incremental
  save + resume is what makes it tolerable. Recommend a `--n-cases` smoke test
  flag (optional) for a first dry run. *(Add only if wanted — not in core.)*
- **JSON reliability:** small local models sometimes wrap JSON in prose; the
  substring-fallback parser handles the common case, but a batch that fails to
  parse should be logged and skipped (its cases retried on a later `--restart`),
  not crash the whole run.
- **`num_ctx` cost:** large context raises memory/latency; `context_window`
  must match what the chosen Ollama model actually supports.
