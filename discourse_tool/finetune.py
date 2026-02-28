import hashlib
import json
import random
from pathlib import Path

import polars as pl

from .config import Config
from .evaluate import create_modelfile, create_ollama_model


# --- Part A: Human Review CLI ---


def _balanced_review_order(to_review):
    """Order segments so every batch of 100 has at least 25 of each binary_flag value."""
    ones = [item for item in to_review if item[1].get("binary_flag") == "1"]
    non_ones = [item for item in to_review if item[1].get("binary_flag") != "1"]
    random.shuffle(ones)
    random.shuffle(non_ones)

    min_per_batch = 25
    batch_size = 100
    result = []

    while len(ones) >= min_per_batch and len(non_ones) >= min_per_batch:
        batch = ones[:min_per_batch] + non_ones[:min_per_batch]
        ones = ones[min_per_batch:]
        non_ones = non_ones[min_per_batch:]

        # Fill remaining slots, drawing from the larger pool to preserve balance
        fill_needed = batch_size - len(batch)
        for _ in range(fill_needed):
            if not ones and not non_ones:
                break
            elif not ones:
                batch.append(non_ones.pop(0))
            elif not non_ones:
                batch.append(ones.pop(0))
            elif len(ones) >= len(non_ones):
                batch.append(ones.pop(0))
            else:
                batch.append(non_ones.pop(0))

        random.shuffle(batch)
        result.extend(batch)

    remaining = ones + non_ones
    if remaining:
        print(
            f"Note: not enough data for balanced batches in last {len(remaining)} "
            f"segments ({len(ones)} flag=1, {len(non_ones)} flag=0 remaining)"
        )
        random.shuffle(remaining)
        result.extend(remaining)

    return result


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

    # Filter to unreviewed segments (skip very short texts) and order with balanced flag distribution
    to_review = [
        (i, row) for i, row in enumerate(df.iter_rows(named=True))
        if (row["source_file"], row["paragraph_index"]) not in reviewed_keys
        and len(row.get("text", "") or "") >= 30
    ]
    to_review = _balanced_review_order(to_review)

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


# --- Scoring ---


def score_model(
    human_labels_path: Path,
    system_prompt_path: Path,
    user_template_path: Path,
    base_model: str = None,
    test_fraction: float = 0.2,
) -> None:
    cfg = Config()
    if base_model is None:
        base_model = cfg.default_model

    df = pl.read_parquet(human_labels_path)
    user_template = user_template_path.read_text(encoding="utf-8").strip()
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()

    # Split by article to avoid data leakage
    articles = df["source_file"].unique().to_list()
    random.shuffle(articles)
    n_test = max(1, int(len(articles) * test_fraction))
    test_articles = set(articles[:n_test])
    train_articles = set(articles[n_test:])

    train_df = df.filter(pl.col("source_file").is_in(train_articles))
    test_df = df.filter(pl.col("source_file").is_in(test_articles))

    print(f"Train: {len(train_articles)} articles ({train_df.height} segments)")
    print(f"Test:  {len(test_articles)} articles ({test_df.height} segments)")

    if train_df.height == 0:
        raise ValueError("No training segments — increase data or decrease test_fraction")
    if test_df.height == 0:
        raise ValueError("No test segments — increase data or increase test_fraction")

    # Build few-shot model from train split
    examples = []
    for row in train_df.iter_rows(named=True):
        example_input = user_template.replace("{text}", row["text"])
        example_output = f"{row['human_flag']} ||| {row.get('position', '')} ||| {row.get('reason', '')}"
        examples.append(f"Example input:\n{example_input}\n\nExample output:\n{example_output}")

    examples_block = "\n\n---\n\n".join(examples)
    enriched_prompt = (
        f"{system_prompt}\n\n"
        f"## Examples of correct evaluations\n\n"
        f"{examples_block}"
    )

    prompt_hash = hashlib.md5(enriched_prompt.encode()).hexdigest()[:8]
    custom_model_name = f"{base_model}-discourse-score-{prompt_hash}"
    modelfile_path = cfg.training_dir / "Modelfile.score"
    cfg.training_dir.mkdir(parents=True, exist_ok=True)
    create_modelfile(base_model, enriched_prompt, modelfile_path)
    create_ollama_model(custom_model_name, modelfile_path)

    # Evaluate test segments
    import ollama
    from tqdm import tqdm

    tp = fp = tn = fn = 0
    for row in tqdm(test_df.iter_rows(named=True), total=test_df.height, desc="Scoring test set"):
        user_message = user_template.replace("{text}", row["text"])
        response = ollama.chat(
            model=custom_model_name,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response["message"]["content"]
        parts = [p.strip() for p in raw.split("|||")]
        predicted = parts[0] if len(parts) >= 1 else None
        actual = str(row["human_flag"])

        if predicted == "1" and actual == "1":
            tp += 1
        elif predicted == "1" and actual != "1":
            fp += 1
        elif predicted != "1" and actual == "1":
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n--- Results ({total} test segments) ---")
    print(f"TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
