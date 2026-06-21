"""Assisted intermediary evaluation — a second-pass LLM "methodological evaluator".

Independently re-assesses the analyst model's initial coding. For each flagged
case (binary_flag=1) it judges whether the assigned label (binary_flag +
position, the analyst's `reason` deliberately excluded) is supported by the
source text, given the original analytical framework.

Cases are sent to the evaluator in batches so it can calibrate CLEAR /
BORDERLINE / ABSENT boundaries comparatively. Batches carry overlap "anchor"
cases from the previous batch so cross-batch consistency and calibration drift
can be measured.

Unlike the rest of the pipeline (the `|||` single-response protocol), the
evaluator returns a JSON array per batch — a deliberate exception. Batched calls
MUST set num_ctx, since Ollama defaults to 2048 tokens and would silently
truncate a batch otherwise.
"""

import hashlib
import json
import random
from pathlib import Path

import ollama
import polars as pl
from tqdm import tqdm

from .config import Config
from .evaluate import create_modelfile, create_ollama_model

# Evaluator columns appended to each case
EVAL_COLS = ["tier", "confidence", "evidence", "strongest_feature", "weakest_feature"]
_EVAL_SCHEMA = {
    "source_file": pl.Utf8,
    "paragraph_index": pl.Int64,
    "tier": pl.Utf8,
    "confidence": pl.Int64,
    "evidence": pl.Utf8,
    "strongest_feature": pl.Utf8,
    "weakest_feature": pl.Utf8,
}


def load_evaluations(path: Path) -> pl.DataFrame:
    """Load an evaluations file (output of `evaluate`) as CSV or parquet.

    The analyst's `|||`-parsed columns can contain malformed model output (e.g.
    a markdown code fence captured into `binary_flag`), so the parsed text
    columns are read as strings rather than letting polars infer numeric dtypes
    and choke on a stray value.
    """
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pl.read_parquet(path)
    elif ext == ".csv":
        text_cols = ("binary_flag", "position", "reason", "raw_evaluation")
        header = pl.read_csv(path, n_rows=0).columns
        overrides = {c: pl.Utf8 for c in text_cols if c in header}
        return pl.read_csv(path, schema_overrides=overrides, infer_schema_length=10000)
    raise ValueError(f"Unsupported evaluations format: {ext} (use .csv or .parquet)")


def select_cases(df: pl.DataFrame, positions: list[str] = None, min_len: int = 30) -> pl.DataFrame:
    """Filter to flagged positives worth re-evaluating.

    Keeps rows with binary_flag=1 and text >= min_len chars. If `positions` is
    given, keeps only positives whose position is in that allow-list.
    """
    # Normalize binary_flag: strip markdown fences/backticks and whitespace that
    # can leak in from malformed analyst output (e.g. "```\n1" -> "1").
    flag = (
        pl.col("binary_flag")
        .cast(pl.Utf8, strict=False)
        .str.replace_all("`", "")
        .str.strip_chars()
    )
    out = df.filter(flag == "1")
    out = out.filter(pl.col("text").is_not_null() & (pl.col("text").str.len_chars() >= min_len))
    if positions:
        out = out.filter(pl.col("position").is_in(list(positions)))
    return out


def build_label(row: dict) -> str:
    """The label the evaluator independently verifies — no analyst reason."""
    position = row.get("position") or ""
    flag = str(row["binary_flag"]).replace("`", "").strip()
    return f"binary_flag={flag}; position={position}"


def build_batch_items(batch_ids: list[int], row_lookup: dict) -> str:
    """Format a batch of cases into the XML structure expected by the evaluator."""
    items = []
    for cid in batch_ids:
        row = row_lookup[cid]
        item = (
            f'<case id="{cid}">\n'
            f'<text>\n{row["text"]}\n</text>\n'
            f'<assigned_label>\n{build_label(row)}\n</assigned_label>\n'
            f'</case>'
        )
        items.append(item)
    return "\n\n".join(items)


def build_evaluator_user_message(
    batch_items: str,
    analyst_system: str,
    analyst_user: str,
    evaluator_user_template: str,
) -> str:
    """Assemble the full evaluator user message for one batch."""
    msg = evaluator_user_template
    msg = msg.replace("{system_prompt}", analyst_system)
    msg = msg.replace("{user_template}", analyst_user)
    msg = msg.replace("{batch_items}", batch_items)
    return msg


def estimate_batch_size(cases: pl.DataFrame, context_window: int) -> int:
    """Estimate how many cases fit in one batch given the model context limit."""
    sample = cases.head(min(10, cases.height))
    total = 0.0
    n = 0
    for row in sample.iter_rows(named=True):
        text_tokens = len(str(row["text"]).split()) * 1.3
        label_tokens = len(build_label(row).split()) * 1.3
        total += text_tokens + label_tokens
        n += 1
    if n == 0:
        return 10
    avg_per_case = total / n
    # Reserve for evaluator system prompt + analyst prompts in the template + output
    available = context_window - 2500
    per_case_total = avg_per_case + 150
    size = int(available / per_case_total) if per_case_total > 0 else 10
    return max(10, min(50, size))


def create_batches(
    case_ids: list[int],
    batch_size: int,
    overlap: int = 5,
    shuffle: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Create batches with optional shuffling and overlap anchors.

    Each batch dict has 'ids' (all case_ids in the batch, anchors first),
    'anchor_ids' (carried from the previous batch) and 'new_ids' (unique to
    this batch). The tail of each batch's new cases anchors the next batch.
    """
    ids = list(case_ids)
    if shuffle:
        random.seed(seed)
        random.shuffle(ids)

    batches = []
    i = 0
    n = len(ids)
    prev_tail = []
    while i < n:
        anchors = prev_tail[-overlap:] if (prev_tail and overlap > 0) else []
        remaining = max(1, batch_size - len(anchors))
        new = ids[i : i + remaining]
        i += remaining
        batches.append({
            "ids": anchors + new,
            "anchor_ids": set(anchors),
            "new_ids": set(new),
        })
        prev_tail = new
    return batches


def parse_evaluator_response(content: str) -> list[dict]:
    """Parse a JSON array from the model response, with substring fallback."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
        raise ValueError(f"Could not parse evaluator response as JSON: {content[:200]}...")


def _normalize_tier(value) -> str:
    tier = (str(value).strip().upper() if value is not None else "") or None
    return tier if tier in ("CLEAR", "BORDERLINE", "ABSENT") else tier


def _coerce_confidence(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_evaluated(all_cases: pl.DataFrame, eval_rows: list[dict]) -> pl.DataFrame:
    """Join accumulated evaluator results onto the full selected-case universe."""
    eval_df = pl.DataFrame(eval_rows) if eval_rows else pl.DataFrame(schema=_EVAL_SCHEMA)
    return all_cases.join(eval_df, on=["source_file", "paragraph_index"], how="left")


def _write_filtered(evaluated: pl.DataFrame, threshold: str, min_confidence: int,
                    filtered_path: Path, filtered_csv: Path) -> pl.DataFrame:
    allowed = ["CLEAR"] if threshold == "clear" else ["CLEAR", "BORDERLINE"]
    filtered = evaluated.filter(
        pl.col("tier").is_in(allowed) & (pl.col("confidence") >= min_confidence)
    )
    filtered.write_parquet(filtered_path)
    filtered.write_csv(filtered_csv)
    return filtered


def calibration_report(batches: list[dict], new_results: dict) -> pl.DataFrame:
    """Per-batch CLEAR/BORDERLINE/ABSENT distribution with drift flags.

    Each batch is scored over the cases it is responsible for (its new_ids),
    so drift in the evaluator's thresholds across batches is visible.
    """
    rows = []
    for batch_idx, batch in enumerate(batches):
        tiers = [new_results[cid]["tier"] for cid in batch["new_ids"] if cid in new_results]
        confs = [new_results[cid]["confidence"] for cid in batch["new_ids"]
                 if cid in new_results and new_results[cid]["confidence"] is not None]
        total = len(tiers)
        if total == 0:
            continue
        rows.append({
            "batch": batch_idx,
            "n_cases": total,
            "pct_clear": round(tiers.count("CLEAR") / total, 3),
            "pct_borderline": round(tiers.count("BORDERLINE") / total, 3),
            "pct_absent": round(tiers.count("ABSENT") / total, 3),
            "avg_confidence": round(sum(confs) / len(confs), 2) if confs else None,
        })

    report = pl.DataFrame(rows)
    if report.height > 1:
        overall_clear = report["pct_clear"].mean()
        overall_border = report["pct_borderline"].mean()
        report = report.with_columns([
            (pl.col("pct_clear") - overall_clear).abs().alias("clear_deviation"),
            (pl.col("pct_borderline") - overall_border).abs().alias("borderline_deviation"),
        ])
        report = report.with_columns(
            ((pl.col("clear_deviation") > 0.20) | (pl.col("borderline_deviation") > 0.20))
            .alias("drift_flag")
        )
    return report


def anchor_consistency(anchor_classifications: dict, new_results: dict) -> pl.DataFrame:
    """Compare each anchor's original evaluation against its re-evaluation(s)."""
    rows = []
    for cid, entries in anchor_classifications.items():
        anchor_tiers = [e["tier"] for e in entries]
        original = new_results.get(cid, {}).get("tier")
        all_tiers = ([original] if original else []) + anchor_tiers
        rows.append({
            "case_id": cid,
            "original_tier": original,
            "reeval_tiers": anchor_tiers,
            "n_evaluations": len(all_tiers),
            "consistent": len(set(t for t in all_tiers if t is not None)) <= 1,
            "batches": [e["batch"] for e in entries],
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def intermediary_evaluate(
    evaluations_path: Path,
    analyst_system_prompt_path: Path,
    analyst_user_template_path: Path,
    evaluator_system_prompt_path: Path,
    evaluator_user_template_path: Path,
    base_model: str = None,
    positions: list[str] = None,
    batch_size: int = None,
    context_window: int = None,
    overlap: int = None,
    shuffle: bool = True,
    seed: int = None,
    threshold: str = "clear",
    min_confidence: int = 1,
    output_dir: Path = None,
    restart: bool = False,
) -> None:
    cfg = Config()
    base_model = base_model or cfg.default_model
    context_window = context_window or cfg.context_window
    overlap = cfg.evaluator_overlap if overlap is None else overlap
    seed = cfg.evaluator_seed if seed is None else seed
    output_dir = output_dir or cfg.intermediary_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data and prompts
    df = load_evaluations(evaluations_path)
    analyst_system = analyst_system_prompt_path.read_text(encoding="utf-8").strip()
    analyst_user = analyst_user_template_path.read_text(encoding="utf-8").strip()
    evaluator_system = evaluator_system_prompt_path.read_text(encoding="utf-8").strip()
    evaluator_user_template = evaluator_user_template_path.read_text(encoding="utf-8")

    # The full universe of cases we care about (positives, length, position filter)
    all_cases = select_cases(df, positions)
    if all_cases.height == 0:
        print("No cases to evaluate (no binary_flag=1 rows matching filters).")
        return

    stem = evaluations_path.stem
    evaluated_path = output_dir / f"{stem}_evaluated.parquet"
    evaluated_csv = output_dir / f"{stem}_evaluated.csv"
    filtered_path = output_dir / f"{stem}_filtered.parquet"
    filtered_csv = output_dir / f"{stem}_filtered.csv"
    calibration_path = output_dir / f"{stem}_calibration.parquet"
    anchors_path = output_dir / f"{stem}_anchors.parquet"

    # Resume: seed evaluator results from a prior run's output
    eval_rows = []
    done_keys = set()
    if not restart and evaluated_path.exists():
        prev = pl.read_parquet(evaluated_path).filter(pl.col("tier").is_not_null())
        eval_rows = prev.select(list(_EVAL_SCHEMA.keys())).to_dicts()
        done_keys = {(r["source_file"], r["paragraph_index"]) for r in eval_rows}

    # Cases still needing evaluation this run
    to_eval = all_cases
    if done_keys:
        keys = list(zip(all_cases["source_file"].to_list(), all_cases["paragraph_index"].to_list()))
        to_eval = all_cases.filter(pl.Series([k not in done_keys for k in keys]))
    to_eval = to_eval.with_row_index("case_id")

    print(f"{len(done_keys)} cases already evaluated, {to_eval.height} to evaluate this run, "
          f"{all_cases.height} selected total")

    if to_eval.height == 0:
        print("Nothing new to evaluate.")
        evaluated = _build_evaluated(all_cases, eval_rows)
        evaluated.write_parquet(evaluated_path)
        evaluated.write_csv(evaluated_csv)
        _write_filtered(evaluated, threshold, min_confidence, filtered_path, filtered_csv)
        return

    # Build the evaluator model (system prompt baked in, content-addressed by hash)
    prompt_hash = hashlib.md5(evaluator_system.encode()).hexdigest()[:8]
    custom_model_name = f"{base_model}-evaluator-{prompt_hash}"
    modelfile_path = output_dir / "Modelfile.evaluator"
    create_modelfile(base_model, evaluator_system, modelfile_path)
    create_ollama_model(custom_model_name, modelfile_path)

    # Batching
    if batch_size is None:
        batch_size = estimate_batch_size(to_eval, context_window)
    if overlap >= batch_size:
        overlap = max(0, batch_size - 1)
    print(f"Batch size: {batch_size}, overlap: {overlap}, context window: {context_window}")

    row_lookup = {row["case_id"]: row for row in to_eval.iter_rows(named=True)}
    batches = create_batches(list(row_lookup.keys()), batch_size, overlap, shuffle, seed)
    print(f"Created {len(batches)} batches")

    new_results = {}            # case_id -> {tier, confidence} (first appearance)
    anchor_classifications = {}  # case_id -> [{batch, tier, confidence}] (anchor re-evals)
    seen_ids = set()
    n_failed = 0

    def _save():
        evaluated = _build_evaluated(all_cases, eval_rows)
        evaluated.write_parquet(evaluated_path)
        evaluated.write_csv(evaluated_csv)

    for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating batches")):
        batch_items = build_batch_items(batch["ids"], row_lookup)
        user_message = build_evaluator_user_message(
            batch_items, analyst_system, analyst_user, evaluator_user_template
        )
        try:
            response = ollama.chat(
                model=custom_model_name,
                messages=[{"role": "user", "content": user_message}],
                options={"num_ctx": context_window},
            )
            evaluations = parse_evaluator_response(response["message"]["content"])
        except Exception as e:
            n_failed += 1
            print(f"\nBatch {batch_idx} failed ({e}); skipping — its cases retry on a later run.")
            continue

        for ev in evaluations:
            try:
                cid = int(ev["case_id"])
            except (KeyError, ValueError, TypeError):
                continue
            if cid not in row_lookup:
                continue
            tier = _normalize_tier(ev.get("tier"))
            conf = _coerce_confidence(ev.get("confidence"))

            if cid in batch["anchor_ids"]:
                anchor_classifications.setdefault(cid, []).append(
                    {"batch": batch_idx, "tier": tier, "confidence": conf}
                )

            if cid not in seen_ids:
                row = row_lookup[cid]
                eval_rows.append({
                    "source_file": row["source_file"],
                    "paragraph_index": row["paragraph_index"],
                    "tier": tier,
                    "confidence": conf,
                    "evidence": ev.get("evidence"),
                    "strongest_feature": ev.get("strongest_feature"),
                    "weakest_feature": ev.get("weakest_feature"),
                })
                new_results[cid] = {"tier": tier, "confidence": conf}
                seen_ids.add(cid)

        _save()

    # Final outputs
    evaluated = _build_evaluated(all_cases, eval_rows)
    evaluated.write_parquet(evaluated_path)
    evaluated.write_csv(evaluated_csv)

    cal_report = calibration_report(batches, new_results)
    anchor_report = anchor_consistency(anchor_classifications, new_results)
    if cal_report.height > 0:
        cal_report.write_parquet(calibration_path)
    if anchor_report.height > 0:
        anchor_report.write_parquet(anchors_path)

    filtered = _write_filtered(evaluated, threshold, min_confidence, filtered_path, filtered_csv)

    # Diagnostics
    print("\n--- Calibration report ---")
    print(cal_report)
    if "drift_flag" in cal_report.columns:
        drifted = cal_report.filter(pl.col("drift_flag"))
        if drifted.height > 0:
            print(f"\nWARNING: {drifted.height} batch(es) show classification drift "
                  f"(>20pp from the mean). Consider larger batches or re-running.")

    if anchor_report.height > 0:
        inconsistent = anchor_report.filter(~pl.col("consistent"))
        print(f"\n--- Anchor consistency ---")
        print(f"Anchor cases: {anchor_report.height}, inconsistent: {inconsistent.height}")
        if inconsistent.height > 0:
            print(inconsistent)

    if n_failed > 0:
        print(f"\n{n_failed} batch(es) failed to parse and were skipped — re-run to retry them.")

    n_eval = evaluated.filter(pl.col("tier").is_not_null()).height
    print(f"\nEvaluated: {n_eval}/{all_cases.height} cases -> {evaluated_path}")
    print(f"Filtered (threshold={threshold}, min_confidence={min_confidence}): "
          f"{filtered.height} cases -> {filtered_path}")
