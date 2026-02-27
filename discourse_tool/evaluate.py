import hashlib
import json
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

    # Create custom model with system prompt baked in
    prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:8]
    custom_model_name = f"{base_model}-discourse-{prompt_hash}"
    modelfile_path = output_dir / "Modelfile"
    create_modelfile(base_model, system_prompt, modelfile_path)
    create_ollama_model(custom_model_name, modelfile_path)

    # Evaluate each paragraph
    total = sum(len(paras) for paras in segments.values())
    rows = []
    with tqdm(total=total, desc="Evaluating paragraphs") as pbar:
        for source_file, paragraphs in segments.items():
            for i, text in enumerate(paragraphs):
                user_message = user_template.replace("{text}", text)

                response = ollama.chat(
                    model=custom_model_name,
                    messages=[{"role": "user", "content": user_message}],
                )
                evaluation = response["message"]["content"]

                rows.append({
                    "source_file": source_file,
                    "paragraph_index": i,
                    "text": text,
                    "model_evaluation": evaluation,
                })
                pbar.set_postfix(file=source_file)
                pbar.update(1)

    # Save results
    df = pl.DataFrame(rows)
    stem = segments_path.stem
    parquet_path = output_dir / f"{stem}.parquet"
    csv_path = output_dir / f"{stem}.csv"
    df.write_parquet(parquet_path)
    df.write_csv(csv_path)
    print(f"Evaluations saved to {parquet_path} and {csv_path}")
