from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Discourse analysis tool using Ollama and open-source LLMs.")


@app.command()
def segment(
    input: Path = typer.Option(..., "--input", "-i", help="Input file, directory (PDF/DOCX/TXT), or CSV file"),
    output: Path = typer.Option("data/segments", "--output", "-o", help="Output directory for segment JSON files"),
    threshold: float = typer.Option(0.3, "--threshold", "-t", help="Cosine similarity threshold for splitting"),
    id_column: Optional[str] = typer.Option(None, "--id-column", help="CSV column name for article identifier"),
    text_column: Optional[str] = typer.Option(None, "--text-column", help="CSV column name for article content"),
    n_files: Optional[int] = typer.Option(None, "--n-files", "-n", help="Only process the first N files/rows (for testing)"),
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model", help="Sentence-transformers model for embeddings (default: all-MiniLM-L6-v2)"),
) -> None:
    """Segment documents into semantically coherent paragraphs.

    For CSV input, provide --id-column and --text-column to specify which
    columns contain the article identifier and text content.
    """
    if input.suffix.lower() == ".csv":
        from .segment import segment_csv

        if not id_column or not text_column:
            raise typer.BadParameter("CSV input requires --id-column and --text-column")
        segment_csv(input, id_column, text_column, output, threshold, n_files, embedding_model)
    else:
        from .segment import segment_files

        segment_files(input, output, threshold, n_files, embedding_model)


@app.command()
def evaluate(
    segments: Path = typer.Option(..., "--segments", "-s", help="Path to segments JSON file"),
    system_prompt: Path = typer.Option(..., "--system-prompt", help="Path to system prompt text file"),
    user_template: Path = typer.Option(..., "--user-template", help="Path to user template text file with {text} placeholder"),
    model: str = typer.Option("mistral", "--model", "-m", help="Base Ollama model name"),
    output: Path = typer.Option("data/evaluations", "--output", "-o", help="Output directory for evaluation results"),
    n_segments: Optional[int] = typer.Option(None, "--n-segments", "-n", help="Only evaluate the first N segments (for testing)"),
) -> None:
    """Evaluate segmented paragraphs using an Ollama model."""
    from .evaluate import evaluate_segments

    evaluate_segments(segments, system_prompt, user_template, model, output, n_segments)


@app.command()
def review(
    evaluations: Path = typer.Option(..., "--evaluations", "-e", help="Path to evaluations parquet file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for human evaluations parquet"),
) -> None:
    """Interactively review and correct model evaluations."""
    from .finetune import review_evaluations

    review_evaluations(evaluations, output)


@app.command()
def finetune(
    human_labels: Path = typer.Option(..., "--human-labels", help="Path to human evaluations parquet"),
    mode: str = typer.Option("few-shot", "--mode", help="Fine-tuning mode: 'few-shot' or 'full'"),
    system_prompt: Optional[Path] = typer.Option(None, "--system-prompt", help="Path to system prompt text file"),
    user_template: Optional[Path] = typer.Option(None, "--user-template", help="Path to user template text file"),
    model: str = typer.Option("mistral", "--model", "-m", help="Base Ollama model name"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
) -> None:
    """Fine-tune or improve the model using human-corrected evaluations."""
    from .finetune import finetune as finetune_fn

    finetune_fn(human_labels, mode, system_prompt, user_template, model, output)


if __name__ == "__main__":
    app()
