# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python CLI research tool for analyzing discourse in large text documents using Ollama and open-source LLMs. Three core modules: semantic document segmentation, discourse evaluation, and fine-tuning from human feedback. Backend-only, no frontend, no API server.

See `implementation_plan.md` for full specification.

## Key Design Constraints

- **Polars only** — no pandas anywhere
- **No async** — sequential processing throughout
- **No database** — flat files only (JSON, parquet, CSV)
- **CLI via Typer** — no web server or API
- **Ollama** — `ollama` Python library for chat/inference, `subprocess` for `ollama create` (Modelfile creation)
- **Sentence embeddings** — `sentence-transformers` with `all-MiniLM-L6-v2` model
- Research tool simplicity — avoid over-engineering

## Architecture

```
discourse_tool/
├── config.py       # Shared config as simple dataclass/dict (paths, model names, thresholds)
├── segment.py      # Document parsing (PDF/TXT/DOCX) & semantic splitting via sentence embeddings
├── evaluate.py     # Ollama Modelfile generation & paragraph-level discourse evaluation
├── finetune.py     # Part A: interactive human review CLI; Part B: few-shot or full fine-tune export
├── cli.py          # Typer CLI entrypoint wiring all modules together
└── data/
    ├── segments/       # JSON output from segmentation
    ├── evaluations/    # Parquet + CSV evaluation results
    └── training/       # Human-labeled data for fine-tuning
```

**Data flow:** Documents → `segment` (JSON paragraphs) → `evaluate` (parquet evaluations) → `review` (human corrections) → `finetune` (improved model or training data export)

## CLI Commands

```bash
python cli.py segment --input paper.pdf --output data/segments/ --threshold 0.3
python cli.py evaluate --segments data/segments/paper.json --system-prompt prompts/system.txt --user-template prompts/template.txt --model mistral --output data/evaluations/
python cli.py review --evaluations data/evaluations/paper.parquet
python cli.py finetune --human-labels data/training/human_evaluations.parquet --mode few-shot --system-prompt prompts/system.txt --model mistral
```

## Dependencies

ollama, sentence-transformers, nltk, pymupdf, python-docx, polars, typer

## Build Order (from implementation plan)

config.py → segment.py → evaluate.py → finetune.py (Part A) → finetune.py (Part B) → cli.py → requirements.txt
