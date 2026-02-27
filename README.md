# Discourse Analysis Tool

A Python CLI tool for analyzing discourse in text documents using Ollama and open-source LLMs. It segments documents into semantically coherent paragraphs, evaluates them against a user-defined theoretical framework, and supports iterative improvement through human feedback.

## Prerequisites

- **Python 3.9+**

### Installing Ollama

On macOS:

```bash
brew install ollama
```

On Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

On Windows, download the installer from https://ollama.com/download.

Start the Ollama server (runs in the background):

```bash
ollama serve
```

The base model (e.g. `mistral`, `llama3`, `gemma2`) is downloaded automatically when you first run the `evaluate` command. Pass the name via `--model`.

## Setup

```bash
pip install -r requirements.txt
```

On first run of the `segment` command, the sentence-transformers model (`all-MiniLM-L6-v2`) and NLTK tokenizer data will be downloaded automatically.

## Preparing Your Prompt Files

Before running evaluations, create two text files:

### System prompt

A text file containing the theoretical framing for your analysis. This becomes the model's system instruction. For example (`prompts/system.txt`):

```
You are a discourse analyst trained in Foucauldian discourse analysis.
Analyze texts for power relations, subject positions, and discursive
formations. Identify how knowledge claims are constructed and legitimized.
```

### User template

A text file containing the per-paragraph evaluation instruction. Use `{text}` as the placeholder where the paragraph text will be inserted. For example (`prompts/template.txt`):

```
Analyze the following paragraph for discursive strategies, identifying
key rhetorical moves, subject positions, and power dynamics:

{text}
```

## Workflow

The tool follows a four-step pipeline:

### 1. Segment documents

Split documents into semantically coherent paragraphs using sentence embeddings.

**From individual files** (PDF, DOCX, TXT):

```bash
python cli.py segment \
  --input paper.pdf \
  --output data/segments/ \
  --threshold 0.3
```

**From a CSV file** with article identifier and content columns:

```bash
python cli.py segment \
  --input articles.csv \
  --id-column article_id \
  --text-column content \
  --output data/segments/ \
  --threshold 0.3
```

- `--input` / `-i`: A single file, directory of files (PDF/DOCX/TXT), or a CSV file
- `--output` / `-o`: Output directory for JSON segment files (default: `data/segments/`)
- `--threshold` / `-t`: Cosine similarity threshold — lower values produce more splits (default: 0.3)
- `--id-column`: (CSV only) Column name for the article identifier
- `--text-column`: (CSV only) Column name for the article text content
- `--n-files` / `-n`: Only process the first N files/rows, for testing on smaller samples
- `--embedding-model`: Sentence-transformers model for computing embeddings (default: `all-MiniLM-L6-v2`)

**Recommended embedding models for Danish text** (per the [Scandinavian Embedding Benchmark](https://huggingface.co/collections/danish-foundation-models/state-of-the-art-danish-models)):

| Model | Size | Notes |
|-------|------|-------|
| `intfloat/multilingual-e5-large` | 0.6B | Best quality, no instructions needed |
| `intfloat/multilingual-e5-base` | 0.3B | Good balance of quality and speed |
| `intfloat/multilingual-e5-small` | ~100M | Lightweight, still solid |
| `KennethTM/MiniLM-L6-danish-encoder-v2` | 22M | Danish-specific, fastest |

Output: JSON file in the format `{"article_id": ["paragraph 1", "paragraph 2", ...]}`.

### 2. Evaluate segments

Run each paragraph through an Ollama model with your system prompt and user template.

```bash
python cli.py evaluate \
  --segments data/segments/paper.json \
  --system-prompt prompts/system.txt \
  --user-template prompts/template.txt \
  --model mistral \
  --output data/evaluations/ \
  --n-articles 5
```

- `--segments` / `-s`: Path to a segments JSON file from step 1
- `--system-prompt`: Path to your system prompt text file
- `--user-template`: Path to your user template text file (must contain `{text}`)
- `--model` / `-m`: Ollama base model name (default: `mistral`)
- `--output` / `-o`: Output directory (default: `data/evaluations/`)
- `--n-articles` / `-n`: Randomly sample N articles to evaluate (for testing)
- `--restart`: Re-evaluate from scratch, ignoring any existing results

**Incremental evaluation:** Re-running the command appends new results — articles that have already been evaluated are automatically skipped. Use `--restart` to discard previous results and start fresh.

A progress bar shows how many paragraphs have been evaluated across all source files.

Output: a `.parquet` and `.csv` file with columns `source_file`, `paragraph_index`, `text`, `binary_flag`, `position`, `reason`, `raw_evaluation`. The three structured columns are parsed from the model response using `|||` as a delimiter. If parsing fails, they are null and the full response is preserved in `raw_evaluation`.

### 3. Review evaluations

Interactively review model outputs and provide corrections.

```bash
python cli.py review \
  --evaluations data/evaluations/paper.parquet
```

- `--evaluations` / `-e`: Path to the evaluation parquet file from step 2
- `--output` / `-o`: Output path for corrected data (default: `data/training/human_evaluations.parquet`)

For each paragraph, you'll see the text and the model's evaluation. Press Enter to accept, or type a corrected evaluation.

### 4. Fine-tune

Use human corrections to improve model performance.

**Few-shot mode** (default) — injects human-corrected examples into an enriched system prompt and creates a new Ollama model:

```bash
python cli.py finetune \
  --human-labels data/training/human_evaluations.parquet \
  --mode few-shot \
  --system-prompt prompts/system.txt \
  --user-template prompts/template.txt \
  --model mistral
```

**Full mode** — exports training data as JSONL and generates an axolotl config for external fine-tuning:

```bash
python cli.py finetune \
  --human-labels data/training/human_evaluations.parquet \
  --mode full \
  --system-prompt prompts/system.txt \
  --user-template prompts/template.txt \
  --output data/training/
```

After full fine-tuning, you import the resulting GGUF weights back into Ollama.

## Project Structure

```
cli.py                  # Top-level entrypoint (run this)
discourse_tool/
├── config.py           # Shared configuration (paths, model defaults, thresholds)
├── segment.py          # Document parsing and semantic splitting
├── evaluate.py         # Ollama model creation and evaluation loop
├── finetune.py         # Human review CLI and fine-tuning data export
├── cli.py              # Typer app definition
data/
├── segments/       # JSON output from segmentation
├── evaluations/    # Parquet + CSV evaluation results
└── training/       # Human-labeled data and fine-tuning artifacts
```
