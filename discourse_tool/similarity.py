import json
from pathlib import Path

import numpy as np
import polars as pl

from .config import Config
from .segment import _init_model_and_nltk


def _cache_paths(source_path: Path) -> tuple[Path, Path]:
    stem = source_path.stem
    parent = source_path.parent
    return (
        parent / f"{stem}_embeddings.npz",
        parent / f"{stem}_embeddings_meta.json",
    )


def _get_or_compute_embeddings(
    texts: list[str],
    source_path: Path,
    model,
    model_name: str,
) -> np.ndarray:
    npz_path, meta_path = _cache_paths(source_path)
    source_mtime = source_path.stat().st_mtime

    # Try loading from cache
    if npz_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("model") == model_name and meta.get("source_mtime") == source_mtime:
            data = np.load(npz_path)
            cached_texts = data["texts"].tolist()
            if cached_texts == texts:
                return data["embeddings"]

    # Compute fresh embeddings
    embeddings = model.encode(texts, show_progress_bar=True)

    # Save cache
    np.savez_compressed(npz_path, embeddings=embeddings, texts=np.array(texts, dtype=object))
    meta_path.write_text(
        json.dumps({"model": model_name, "source_mtime": source_mtime}),
        encoding="utf-8",
    )

    return embeddings


def search_similar(
    evaluations_path: Path,
    target_path: Path,
    embedding_model: str = None,
    top_n: int = 10,
    output_dir: Path = None,
) -> None:
    cfg = Config()
    if embedding_model is None:
        embedding_model = cfg.embedding_model
    if output_dir is None:
        output_dir = cfg.evaluations_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load evaluations and filter to flag=1
    eval_df = pl.read_parquet(evaluations_path)
    positives = eval_df.filter(pl.col("binary_flag") == "1")

    if positives.height == 0:
        print("No segments with binary_flag=1 found in evaluations.")
        return

    # 2. Group by position and print summary
    pos_groups = positives.group_by("position").agg(pl.col("text"))
    position_names = pos_groups["position"].to_list()
    position_texts = pos_groups["text"].to_list()

    summary_parts = [f"{name} ({len(texts)} texts)" for name, texts in zip(position_names, position_texts)]
    print(f"Found {len(position_names)} positions: {', '.join(summary_parts)}")

    # 3. Load embedding model
    model = _init_model_and_nltk(embedding_model)

    # 4. Embed flag=1 texts and compute centroids
    all_pos_texts = positives["text"].to_list()
    pos_embeddings = _get_or_compute_embeddings(
        all_pos_texts, evaluations_path, model, embedding_model,
    )

    # Build centroid per position
    centroids = {}
    offset = 0
    for name, texts in zip(position_names, position_texts):
        n = len(texts)
        centroid = pos_embeddings[offset : offset + n].mean(axis=0)
        centroids[name] = centroid
        offset += n

    # 5. Load target file
    ext = target_path.suffix.lower()
    if ext == ".json":
        segments = json.loads(target_path.read_text(encoding="utf-8"))
        target_rows = [
            (source_file, i, text)
            for source_file, paragraphs in segments.items()
            for i, text in enumerate(paragraphs)
            if len(text) >= 30
        ]
    elif ext == ".parquet":
        target_df = pl.read_parquet(target_path)
        target_rows = [
            (row["source_file"], row["paragraph_index"], row["text"])
            for row in target_df.iter_rows(named=True)
            if len(row["text"]) >= 30
        ]
    else:
        raise ValueError(f"Unsupported target format: {ext} (use .json or .parquet)")

    if not target_rows:
        print("No target segments found (or all < 30 chars).")
        return

    source_files, para_indices, target_texts = zip(*target_rows)

    # 6. Embed target texts
    target_embeddings = _get_or_compute_embeddings(
        list(target_texts), target_path, model, embedding_model,
    )

    # 7. Compute cosine similarity (L2-normalize then matrix multiply)
    centroid_names = list(centroids.keys())
    centroid_matrix = np.array([centroids[n] for n in centroid_names])

    # Normalize
    target_norms = np.linalg.norm(target_embeddings, axis=1, keepdims=True)
    target_norms[target_norms == 0] = 1
    target_normed = target_embeddings / target_norms

    centroid_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
    centroid_norms[centroid_norms == 0] = 1
    centroid_normed = centroid_matrix / centroid_norms

    # [n_targets, n_positions]
    similarities = target_normed @ centroid_normed.T

    # 8. Build results dataframe (one row per target segment per position)
    rows = []
    for i in range(len(target_texts)):
        for j, pos_name in enumerate(centroid_names):
            rows.append({
                "source_file": source_files[i],
                "paragraph_index": para_indices[i],
                "text": target_texts[i],
                "position": pos_name,
                "similarity": float(similarities[i, j]),
            })

    results_df = pl.DataFrame(rows)

    # 9. Print top-N per position
    for pos_name in centroid_names:
        pos_df = results_df.filter(pl.col("position") == pos_name).sort("similarity", descending=True)
        top = pos_df.head(top_n)
        print(f"\n{'=' * 60}")
        print(f"Position: {pos_name}")
        print(f"{'=' * 60}")
        for row in top.iter_rows(named=True):
            text_preview = row["text"][:120].replace("\n", " ")
            if len(row["text"]) > 120:
                text_preview += "..."
            print(f"  [{row['similarity']:.3f}] {row['source_file']}:{row['paragraph_index']}")
            print(f"         {text_preview}")

    # 10. Save full results
    results_df = results_df.sort(["position", "similarity"], descending=[False, True])
    out_path = output_dir / f"{target_path.stem}_similarity.parquet"
    results_df.write_parquet(out_path)
    print(f"\nFull results saved to {out_path}")
