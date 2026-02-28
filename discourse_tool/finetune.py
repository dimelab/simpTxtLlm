import json
from pathlib import Path

import polars as pl

from .config import Config
from .evaluate import create_modelfile, create_ollama_model


# --- Part A: Human Review CLI ---


def review_evaluations(evaluations_path: Path, output_path: Path = None) -> None:
    cfg = Config()
    if output_path is None:
        output_path = cfg.training_dir / "human_evaluations.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(evaluations_path)

    # Load already-reviewed rows if output exists
    reviewed_rows = []
    reviewed_keys = set()
    if output_path.exists():
        existing = pl.read_parquet(output_path)
        reviewed_rows = existing.to_dicts()
        reviewed_keys = {
            (r["source_file"], r["paragraph_index"]) for r in reviewed_rows
        }

    # Filter to unreviewed segments
    to_review = [
        (i, row) for i, row in enumerate(df.iter_rows(named=True))
        if (row["source_file"], row["paragraph_index"]) not in reviewed_keys
    ]

    total = len(df)
    n_already = len(reviewed_keys)
    print(f"{n_already}/{total} segments already reviewed, {len(to_review)} remaining")

    if not to_review:
        print("Nothing new to review.")
        return

    for idx, (i, row) in enumerate(to_review):
        model_flag = row.get("binary_flag", "?")
        print(f"\n--- Segment {n_already + idx + 1}/{total} [{row['source_file']} §{row['paragraph_index']}] ---")
        print(f"\nText:\n{row['text']}")
        print(f"\nModel flag: {model_flag}")
        if row.get("position"):
            print(f"Position: {row['position']}")
        if row.get("reason"):
            print(f"Reason: {row['reason']}")
        print(f"\nPress Enter to accept '{model_flag}', or type 0/1 to override:")

        while True:
            user_input = input("> ").strip()
            if user_input == "":
                human_flag = model_flag
                accepted = True
                break
            elif user_input in ("0", "1"):
                human_flag = user_input
                accepted = user_input == model_flag
                break
            else:
                print("Please enter 0, 1, or press Enter to accept.")

        reviewed_row = dict(row)
        reviewed_row["human_flag"] = human_flag
        reviewed_row["accepted"] = accepted
        reviewed_rows.append(reviewed_row)

        # Save after every review so no progress is lost
        pl.DataFrame(reviewed_rows).write_parquet(output_path)

    print(f"\nReview complete. {total} segments saved to {output_path}")


# --- Part B: Fine-Tuning ---


def finetune(
    human_labels_path: Path,
    mode: str = "few-shot",
    system_prompt_path: Path = None,
    user_template_path: Path = None,
    base_model: str = None,
    output_dir: Path = None,
) -> None:
    cfg = Config()
    if base_model is None:
        base_model = cfg.default_model
    if output_dir is None:
        output_dir = cfg.training_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(human_labels_path)

    if mode == "few-shot":
        _finetune_few_shot(df, system_prompt_path, user_template_path, base_model, output_dir)
    elif mode == "full":
        _finetune_full(df, system_prompt_path, user_template_path, base_model, output_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'few-shot' or 'full'.")


def _finetune_few_shot(
    df: pl.DataFrame,
    system_prompt_path: Path,
    user_template_path: Path,
    base_model: str,
    output_dir: Path,
) -> None:
    if system_prompt_path is None or user_template_path is None:
        raise ValueError("few-shot mode requires --system-prompt and --user-template")

    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    user_template = user_template_path.read_text(encoding="utf-8").strip()

    # Build few-shot examples block
    examples = []
    for row in df.iter_rows(named=True):
        example_input = user_template.replace("{text}", row["text"])
        example_output = f"{row['human_flag']} ||| {row.get('position', '')} ||| {row.get('reason', '')}"
        examples.append(f"Example input:\n{example_input}\n\nExample output:\n{example_output}")

    examples_block = "\n\n---\n\n".join(examples)
    enriched_prompt = (
        f"{system_prompt}\n\n"
        f"## Examples of correct evaluations\n\n"
        f"{examples_block}"
    )

    # Create new model with enriched prompt
    import hashlib
    prompt_hash = hashlib.md5(enriched_prompt.encode()).hexdigest()[:8]
    custom_model_name = f"{base_model}-discourse-fewshot-{prompt_hash}"
    modelfile_path = output_dir / "Modelfile.fewshot"
    create_modelfile(base_model, enriched_prompt, modelfile_path)
    create_ollama_model(custom_model_name, modelfile_path)
    print(f"\nFew-shot model created: {custom_model_name}")
    print(f"Use with: ollama run {custom_model_name}")


def _finetune_full(
    df: pl.DataFrame,
    system_prompt_path: Path,
    user_template_path: Path,
    base_model: str,
    output_dir: Path,
) -> None:
    system_prompt = ""
    if system_prompt_path is not None:
        system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()

    user_template = "{text}"
    if user_template_path is not None:
        user_template = user_template_path.read_text(encoding="utf-8").strip()

    # Export training data as JSONL
    jsonl_path = output_dir / "training_data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in df.iter_rows(named=True):
            user_message = user_template.replace("{text}", row["text"])
            if system_prompt:
                prompt = f"{system_prompt}\n\n{user_message}"
            else:
                prompt = user_message
            response = f"{row['human_flag']} ||| {row.get('position', '')} ||| {row.get('reason', '')}"
            entry = {"prompt": prompt, "response": response}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Generate axolotl config
    config_path = output_dir / "axolotl_config.yml"
    axolotl_config = f"""base_model: {base_model}
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

datasets:
  - path: {jsonl_path}
    type: completion

sequence_len: 2048
micro_batch_size: 1
gradient_accumulation_steps: 4
num_epochs: 3
learning_rate: 2e-5
optimizer: adamw_torch
lr_scheduler: cosine

output_dir: {output_dir / "output"}
"""
    config_path.write_text(axolotl_config, encoding="utf-8")

    print(f"\nTraining data exported to {jsonl_path} ({len(df)} examples)")
    print(f"Axolotl config written to {config_path}")
    print(f"\nTo fine-tune, run:")
    print(f"  accelerate launch -m axolotl.cli.train {config_path}")
    print(f"\nAfter training, convert to GGUF and import into Ollama:")
    print(f"  ollama create {base_model}-finetuned -f Modelfile")
