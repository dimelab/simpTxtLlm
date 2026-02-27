# Discourse Analysis Tool — Implementation Plan

## Overview

A Python CLI tool for analyzing discourse in large text documents using Ollama and open-source LLMs. Three core modules: document segmentation, discourse evaluation, and fine-tuning from human feedback. Backend only, no frontend. Polars over pandas. Keep it simple — this is a research tool, not a production app.

## Project Structure

```
discourse_tool/
├── segment.py          # Document parsing & semantic splitting
├── evaluate.py         # Ollama modelfile generation & discourse evaluation
├── finetune.py         # Fine-tuning data prep & execution
├── config.py           # Shared config (model names, paths, etc.)
├── cli.py              # Simple CLI entrypoint (Typer)
├── requirements.txt
└── data/
    ├── segments/       # Output JSON from segmentation
    ├── evaluations/    # Model evaluation outputs
    └── training/       # Human-labeled data for fine-tuning
```

## Dependencies

```
ollama
sentence-transformers
nltk
pymupdf
python-docx
polars
typer
```

## Module 1: `segment.py` — Semantic Segmentation

**Purpose:** Parse a document (PDF, TXT, DOCX), split it into semantically coherent paragraphs, output JSON keyed by source filename.

**Implementation details:**

- File reading: use `pymupdf` for PDF, `python-docx` for DOCX, plain `open()` for TXT.
- Semantic splitting approach:
  1. Sentence tokenization via `nltk.sent_tokenize`.
  2. Embed each sentence using `sentence-transformers` with the `all-MiniLM-L6-v2` model.
  3. Compute cosine similarity between consecutive sentence embeddings.
  4. Split into a new paragraph wherever similarity drops below a configurable threshold (default 0.3).
- Output format: `{"source_file.pdf": ["paragraph 1 text", "paragraph 2 text", ...]}`.
- The similarity threshold should be a CLI argument.
- Process multiple files if a directory is passed.

**CLI usage:**

```bash
python cli.py segment --input paper.pdf --output data/segments/ --threshold 0.3
```

## Module 2: `evaluate.py` — Discourse Evaluation via Ollama

**Purpose:** Let the user specify a theoretically informed system prompt and a user-level evaluation instruction, create an Ollama model from a Modelfile, then run each segmented paragraph through it and collect evaluations.

**Implementation details:**

- Inputs:
  - `system_prompt`: path to a text file containing the theoretical framing (e.g., Foucauldian discourse analysis, CDA, rhetorical analysis, etc.). This becomes the SYSTEM instruction in the Modelfile.
  - `user_template`: path to a text file containing the evaluation instruction with a `{text}` placeholder where the paragraph text gets inserted.
  - `base_model`: which Ollama model to use as the base (e.g., `mistral`, `llama3`, `gemma2`).
  - `segments_path`: path to the JSON output from Module 1.
- Modelfile generation:
  - Write a Modelfile to disk with the format:
    ```
    FROM <base_model>
    SYSTEM """<system_prompt contents>"""
    ```
  - Run `ollama create <custom_model_name> -f Modelfile` via `subprocess.run`.
- Evaluation loop:
  - Load the segments JSON.
  - For each paragraph, format the user template with the paragraph text inserted at `{text}`.
  - Call `ollama.chat()` with the custom model, passing the formatted user message.
  - Collect responses.
- Output: a Polars DataFrame saved as parquet with columns: `source_file`, `paragraph_index`, `text`, `model_evaluation`. Also save as CSV for easy human review.

**CLI usage:**

```bash
python cli.py evaluate \
  --segments data/segments/paper.json \
  --system-prompt prompts/foucault_system.txt \
  --user-template prompts/cda_template.txt \
  --model mistral \
  --output data/evaluations/
```

## Module 3: `finetune.py` — Fine-Tuning from Human Evaluations

**Purpose:** Allow the user to review and correct model evaluations, then use those corrections to improve the model.

### Part A: Human Review CLI

- Load the evaluation parquet from Module 2.
- Interactive CLI loop: for each row, display the paragraph text and the model's evaluation.
- Human either presses Enter to accept or types a corrected evaluation.
- Save results to `human_evaluations.parquet` with an added `human_evaluation` column and a `accepted` boolean column.

**CLI usage:**

```bash
python cli.py review --evaluations data/evaluations/paper.parquet
```

### Part B: Fine-Tuning

Two modes, selectable via flag:

**Mode 1: `few-shot` (default, simpler)**
- Take the human-corrected examples and format them as few-shot examples.
- Inject them into an updated system prompt that includes the theoretical framing plus concrete examples of correct evaluations.
- Regenerate the Modelfile with the enriched system prompt and re-create the Ollama model.
- This doesn't update weights but iteratively improves output quality through in-context learning.

**Mode 2: `full` (actual fine-tuning)**
- Convert human-labeled data to training JSONL format:
  ```json
  {"prompt": "<system + user template + paragraph>", "response": "<human evaluation>"}
  ```
- Export in a format compatible with `unsloth` or `axolotl`.
- Provide the user with the exact commands to run fine-tuning externally (since Ollama doesn't support native fine-tuning).
- After fine-tuning, the user imports the new GGUF weights into Ollama via a Modelfile pointing to the fine-tuned model file.
- The script handles data conversion and generates the necessary config files and shell commands.

**CLI usage:**

```bash
# Few-shot mode
python cli.py finetune \
  --human-labels data/training/human_evaluations.parquet \
  --mode few-shot \
  --system-prompt prompts/foucault_system.txt \
  --model mistral

# Full fine-tune (generates training data + instructions)
python cli.py finetune \
  --human-labels data/training/human_evaluations.parquet \
  --mode full \
  --output data/training/
```

## `config.py` — Shared Configuration

- Default paths for data directories.
- Default model name.
- Default similarity threshold.
- Default Modelfile template string.
- Keep it as a simple dataclass or dict, nothing fancy.

## Design Decisions

- **No async** — sequential processing is fine for a research tool.
- **No database** — flat files (JSON, parquet, CSV) are the data layer.
- **No API/server** — CLI only via Typer.
- **No containerization** — assumes Ollama is installed and running locally.
- **No embedding caching** — add later if segmentation is slow on large corpora.
- **Polars** for all DataFrame operations, no pandas.
- **subprocess** for `ollama create` since the Python library doesn't expose Modelfile creation.
- **`ollama` Python library** for chat/inference calls.

## Build Order

1. `config.py` — set up shared defaults and paths.
2. `segment.py` — get document parsing and semantic splitting working.
3. `evaluate.py` — Modelfile generation and evaluation loop.
4. `finetune.py` Part A — human review CLI.
5. `finetune.py` Part B — few-shot enrichment first, full fine-tune export second.
6. `cli.py` — wire everything together with Typer.
7. `requirements.txt` — pin dependencies.
