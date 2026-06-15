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


@app.command(name="search-similar")
def search_similar_cmd(
    evaluations: Path = typer.Option(..., "--evaluations", "-e", help="Reference parquet with labelled positions (binary_flag=1)"),
    target: Path = typer.Option(..., "--target", "-t", help="Target segments JSON or evaluation parquet to search"),
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model", help="Sentence-transformers model for embeddings (default: all-MiniLM-L6-v2)"),
    top_n: int = typer.Option(10, "--top-n", "-n", help="Number of most similar segments to display per position"),
    top_positions: Optional[int] = typer.Option(None, "--top-positions", help="Only use the N positions with the most texts"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for results (default: data/similarity/)"),
    restart: bool = typer.Option(False, "--restart", help="Recompute embeddings from scratch, ignoring cache"),
) -> None:
    """Find segments similar to known discourse positions using embedding similarity.

    Takes an evaluation parquet with labelled positions and ranks segments in a
    target file by cosine similarity to each position's centroid embedding.
    """
    from .similarity import search_similar

    search_similar(evaluations, target, embedding_model, top_n, top_positions, output, restart)


@app.command(name="normalize-positions")
def normalize_positions_cmd(
    evaluations: Path = typer.Option(..., "--evaluations", "-e", help="Path to evaluations parquet file"),
    positions: list[str] = typer.Option(..., "--position", "-p", help="Canonical position label (repeatable)"),
    threshold: float = typer.Option(0.6, "--threshold", "-t", help="Minimum similarity ratio to accept a match"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output parquet path (default: overwrite in-place)"),
) -> None:
    """Normalize noisy position labels to a canonical list using fuzzy matching.

    Provide the canonical positions with repeated -p flags. Each unique position
    in the evaluations file is matched to the closest canonical label using
    string similarity. Positions below the threshold are kept as-is.
    """
    from .similarity import normalize_positions

    normalize_positions(evaluations, positions, threshold, output)


@app.command(name="intermediary-evaluate")
def intermediary_evaluate_cmd(
    evaluations: Path = typer.Option(..., "--evaluations", "-e", help="Evaluations file (CSV or parquet) from the evaluate step"),
    analyst_system_prompt: Path = typer.Option(..., "--analyst-system-prompt", help="System prompt used for the original evaluation"),
    analyst_user_template: Path = typer.Option(..., "--analyst-user-template", help="User template used for the original evaluation"),
    evaluator_system_prompt: Path = typer.Option(..., "--evaluator-system-prompt", help="Evaluator system prompt (e.g. prompts/evaluator_system.txt)"),
    evaluator_user_template: Path = typer.Option(..., "--evaluator-user-template", help="Evaluator user template (e.g. prompts/evaluator_user_template.txt)"),
    model: str = typer.Option("mistral", "--model", "-m", help="Base Ollama model for the evaluator"),
    position: Optional[list[str]] = typer.Option(None, "--position", "-p", help="Restrict to these positions (repeatable); others discarded"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Cases per batch (default: auto-estimate from context window)"),
    context_window: Optional[int] = typer.Option(None, "--context-window", help="Ollama num_ctx for batched calls (default: 32000)"),
    overlap: Optional[int] = typer.Option(None, "--overlap", help="Anchor cases carried between batches (default: 5)"),
    shuffle: bool = typer.Option(True, "--shuffle/--no-shuffle", help="Shuffle cases before batching"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Shuffle seed (default: 42)"),
    threshold: str = typer.Option("clear", "--threshold", help="Filter tier: 'clear' (CLEAR only) or 'borderline' (CLEAR+BORDERLINE)"),
    min_confidence: int = typer.Option(1, "--min-confidence", help="Minimum evaluator confidence (1-5) to pass the filter"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory (default: data/intermediary/)"),
    restart: bool = typer.Option(False, "--restart", help="Re-evaluate from scratch, ignoring existing results"),
) -> None:
    """Independently re-assess the initial coding of flagged paragraphs.

    A second-pass evaluator model reads each flagged case (binary_flag=1) against
    the original analytical framework and tiers it CLEAR / BORDERLINE / ABSENT.
    Re-running resumes; already-evaluated cases are skipped unless --restart.
    """
    if threshold not in ("clear", "borderline"):
        raise typer.BadParameter("--threshold must be 'clear' or 'borderline'")

    from .intermediary import intermediary_evaluate

    intermediary_evaluate(
        evaluations, analyst_system_prompt, analyst_user_template,
        evaluator_system_prompt, evaluator_user_template,
        base_model=model, positions=position, batch_size=batch_size,
        context_window=context_window, overlap=overlap, shuffle=shuffle,
        seed=seed, threshold=threshold, min_confidence=min_confidence,
        output_dir=output, restart=restart,
    )


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
