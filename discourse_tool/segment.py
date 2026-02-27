import json
from pathlib import Path

import fitz  # pymupdf
import nltk
import numpy as np
import polars as pl
from docx import Document
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import Config


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    elif ext == ".docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".txt":
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def semantic_split(text: str, model: SentenceTransformer, threshold: float) -> list[str]:
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    embeddings = model.encode(sentences)

    # Cosine similarity between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        similarities.append(cos_sim)

    # Split where similarity drops below threshold
    paragraphs = []
    current = [sentences[0]]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            paragraphs.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    paragraphs.append(" ".join(current))

    return paragraphs


def _init_model_and_nltk() -> SentenceTransformer:
    """Load sentence-transformers model and ensure NLTK data is available."""
    cfg = Config()
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    return SentenceTransformer(cfg.embedding_model)


def segment_csv(
    csv_path: Path,
    id_column: str,
    text_column: str,
    output_dir: Path,
    threshold: float = None,
) -> None:
    """Segment articles from a CSV file.

    Reads a CSV with an identifier column and a text column, segments each
    article, and writes a single JSON output file.
    """
    cfg = Config()
    if threshold is None:
        threshold = cfg.similarity_threshold

    output_dir.mkdir(parents=True, exist_ok=True)
    model = _init_model_and_nltk()

    df = pl.read_csv(csv_path)

    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in CSV. Available: {df.columns}")
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV. Available: {df.columns}")

    results = {}
    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Segmenting articles"):
        article_id = str(row[id_column])
        text = str(row[text_column])
        if not text.strip():
            results[article_id] = []
            continue
        paragraphs = semantic_split(text, model, threshold)
        results[article_id] = paragraphs

    out_file = output_dir / f"{csv_path.stem}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(results)} articles segmented -> {out_file}")


def segment_files(input_path: Path, output_dir: Path, threshold: float = None) -> None:
    """Segment individual document files (PDF, DOCX, TXT)."""
    cfg = Config()
    if threshold is None:
        threshold = cfg.similarity_threshold

    output_dir.mkdir(parents=True, exist_ok=True)
    model = _init_model_and_nltk()

    # Handle single file or directory
    if input_path.is_dir():
        files = [f for f in input_path.iterdir() if f.suffix.lower() in (".pdf", ".docx", ".txt")]
    else:
        files = [input_path]

    for file_path in tqdm(files, desc="Segmenting files"):
        text = read_file(file_path)
        paragraphs = semantic_split(text, model, threshold)
        output = {file_path.name: paragraphs}
        out_file = output_dir / f"{file_path.stem}.json"
        out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(files)} files segmented -> {output_dir}")
