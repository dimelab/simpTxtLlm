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
    n_articles: Optional[int] = typer.Option(None, "--n-articles", "-n", help="Randomly sample N articles to evaluate (for testing)"),
    restart: bool = typer.Option(False, "--restart", help="Re-evaluate from scratch, ignoring existing results"),
) -> None:
    """Evaluate segmented paragraphs using an Ollama model.

    Re-running appends results incrementally — already-evaluated articles are
    skipped. Use --restart to force a fresh evaluation.
    """
    from .evaluate import evaluate_segments

    evaluate_segments(segments, system_prompt, user_template, model, output, n_articles, restart)


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


@app.command()
def score(
    human_labels: Path = typer.Option(..., "--human-labels", help="Path to human evaluations parquet"),
    system_prompt: Path = typer.Option(..., "--system-prompt", help="Path to system prompt text file"),
    user_template: Path = typer.Option(..., "--user-template", help="Path to user template text file with {text} placeholder"),
    model: str = typer.Option("mistral", "--model", "-m", help="Base Ollama model name"),
    test_fraction: float = typer.Option(0.2, "--test-fraction", "-t", help="Fraction of articles to hold out for testing"),
) -> None:
    """Evaluate few-shot model accuracy with a train/test split.

    Splits human-labelled data by article into train and test sets, builds a
    few-shot model from the train set, runs it on the test set, and reports
    accuracy, precision, recall, and F1.
    """
    from .finetune import score_model

    score_model(human_labels, system_prompt, user_template, model, test_fraction)


@app.command()
def stats(
    file: Path = typer.Option(..., "--file", "-f", help="Path to evaluations or reviewed parquet file"),
) -> None:
    """Show statistics for an evaluations or reviewed data file."""
    import polars as pl

    df = pl.read_parquet(file)
    n_segments = len(df)
    n_articles = df["source_file"].n_unique()

    print(f"\nFile: {file}")
    print(f"Segments: {n_segments}")
    print(f"Articles: {n_articles}")

    # Binary flag distribution (from evaluation or review)
    flag_col = "human_flag" if "human_flag" in df.columns else "binary_flag"
    if flag_col in df.columns:
        counts = df.group_by(flag_col).len().sort(flag_col)
        print(f"\n{flag_col} distribution:")
        for row in counts.iter_rows(named=True):
            pct = 100 * row["len"] / n_segments
            print(f"  {row[flag_col]}: {row['len']} ({pct:.1f}%)")

    # Acceptance rate (reviewed data only)
    if "accepted" in df.columns:
        n_accepted = df.filter(pl.col("accepted")).height
        print(f"\nAccepted (agreed with model): {n_accepted}/{n_segments} ({100 * n_accepted / n_segments:.1f}%)")

    # Position breakdown (if present and flag is 1)
    if "position" in df.columns:
        positives = df.filter(pl.col(flag_col) == "1") if flag_col in df.columns else df
        if positives.height > 0:
            pos_counts = positives.group_by("position").len().sort("len", descending=True)
            print(f"\nPosition breakdown (flag=1):")
            for row in pos_counts.iter_rows(named=True):
                label = row["position"] if row["position"] else "(empty)"
                print(f"  {label}: {row['len']}")

    print()


if __name__ == "__main__":
    app()
