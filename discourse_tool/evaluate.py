import hashlib
import json
import random
import subprocess
from pathlib import Path

import ollama
import polars as pl
from tqdm import tqdm

from .config import Config


def create_modelfile(base_model: str, system_prompt: str, output_path: Path) -> Path:
    cfg = Config()
    content = cfg.modelfile_template.format(
        base_model=base_model,
        system_prompt=system_prompt,
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path


def create_ollama_model(model_name: str, modelfile_path: Path) -> None:
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ollama create failed: {result.stderr}")
    print(f"Created Ollama model: {model_name}")


def evaluate_segments(
    segments_path: Path,
    system_prompt_path: Path,
    user_template_path: Path,
    base_model: str = None,
    output_dir: Path = None,
    n_articles: int = None,
    restart: bool = False,
) -> None:
    cfg = Config()
    if base_model is None:
        base_model = cfg.default_model
    if output_dir is None:
        output_dir = cfg.evaluations_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    user_template = user_template_path.read_text(encoding="utf-8").strip()

    # Determine output paths
    stem = segments_path.stem
    parquet_path = output_dir / f"{stem}.parquet"
    csv_path = output_dir / f"{stem}.csv"

    # Check for existing results (incremental mode)
    existing_df = None
    already_evaluated = set()
    if not restart and parquet_path.exists():
        existing_df = pl.read_parquet(parquet_path)
        already_evaluated = set(existing_df["source_file"].unique().to_list())

    # Filter to unevaluated articles
    unevaluated = {k: v for k, v in segments.items() if k not in already_evaluated}

    # Random sampling
    article_ids = list(unevaluated.keys())
    if n_articles is not None and len(article_ids) > n_articles:
        article_ids = random.sample(article_ids, n_articles)

    total_articles = len(segments)
    n_already = len(already_evaluated)
    n_this_run = len(article_ids)
    n_paragraphs = sum(
        1 for aid in article_ids
        for text in unevaluated[aid]
        if len(text) >= 30
    )
    print(f"{n_already} articles already evaluated, {n_this_run} to evaluate this run ({n_paragraphs} paragraphs), {total_articles} total")

    if n_this_run == 0:
        print("Nothing new to evaluate.")
        return

    # Create custom model with system prompt baked in
    prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:8]
    custom_model_name = f"{base_model}-discourse-{prompt_hash}"
    modelfile_path = output_dir / "Modelfile"
    create_modelfile(base_model, system_prompt, modelfile_path)
    create_ollama_model(custom_model_name, modelfile_path)

    # Build paragraph list from selected articles, skip very short segments
    all_paragraphs = [
        (article_id, i, text)
        for article_id in article_ids
        for i, text in enumerate(unevaluated[article_id])
        if len(text) >= 30
    ]
    random.shuffle(all_paragraphs)

    # Evaluate each paragraph, saving every 1000 segments
    rows = []
    for source_file, i, text in tqdm(all_paragraphs, desc="Evaluating paragraphs"):
        user_message = user_template.replace("{text}", text)

        response = ollama.chat(
            model=custom_model_name,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response["message"]["content"]

        # Parse structured output: binary_flag ||| position ||| reason
        parts = [p.strip() for p in raw.split("|||")]
        if len(parts) == 3:
            binary_flag = parts[0]
            position = parts[1]
            reason = parts[2]
        else:
            binary_flag = None
            position = None
            reason = None

        rows.append({
            "source_file": source_file,
            "paragraph_index": i,
            "text": text,
            "binary_flag": binary_flag,
            "position": position,
            "reason": reason,
            "raw_evaluation": raw,
        })

        if len(rows) % 1000 == 0:
            new_df = pl.DataFrame(rows)
            if existing_df is not None and not restart:
                df = pl.concat([existing_df, new_df])
            else:
                df = new_df
            df.write_parquet(parquet_path)
            df.write_csv(csv_path)

    # Final save
    new_df = pl.DataFrame(rows)
    if existing_df is not None and not restart:
        df = pl.concat([existing_df, new_df])
    else:
        df = new_df
    df.write_parquet(parquet_path)
    df.write_csv(csv_path)
    print(f"Evaluations saved to {parquet_path} and {csv_path}")
