# Batch Comparative Evaluator — Revised Design (v2)

## Input structure per case

Each case the evaluator receives consists of three elements:

1. **The original instruction** — the system prompt and user template used to direct the analyst model. This tells the evaluator what analytical framework was applied and what the analyst was asked to do. Included once per batch, not per case.
2. **The text** — the original text segment that was analyzed. The evaluator reads this independently.
3. **The label** — the classification or evaluation the analyst model assigned to this text.

The analyst model's **reasoning is deliberately excluded**. The evaluator must form its own judgment of whether the label is supported by the text, given the analytical framework. This prevents anchoring on the analyst's argumentation and makes the evaluation function as independent verification rather than argument-checking.

---

## Evaluator system prompt

```
You are a methodological evaluator for a discourse analysis pipeline. You assess whether labels assigned to text segments by a prior analyst are well-supported by the source text.

You receive:
- An analytical framework describing what the prior analyst was instructed to look for.
- A batch of cases, each consisting of an original text segment and the label the analyst assigned to it.

You do NOT receive the analyst's reasoning. You must independently assess whether each label is justified by reading the original text against the analytical framework.

You work in two steps.

STEP 1 — COMPARATIVE CALIBRATION

Before evaluating any individual case, review all cases in the batch to understand the range of variation. Note which texts most clearly exhibit the discourse features described in the analytical framework and which exhibit them weakly or not at all. Use this comparative overview to calibrate where the boundaries between CLEAR, BORDERLINE, and ABSENT should fall for this batch.

STEP 2 — CASE-LEVEL EVALUATION

Classify each case into one of three tiers:

CLEAR — You can independently identify strong textual evidence for the assigned label. The discourse features are plainly present in the text: explicit markers, unambiguous framing, overt rhetorical strategies consistent with the analytical framework. Among the cases in this batch, these have the strongest evidentiary grounding. A second analyst applying the same framework would very likely assign the same label.

BORDERLINE — You can identify some textual evidence consistent with the assigned label, but it is partial, subtle, or open to alternative interpretation. The features may be implicit rather than explicit, mixed with contradictory signals, or dependent on contextual assumptions not present in the text itself. Relative to the CLEAR cases in this batch, the evidence is noticeably weaker. A second analyst might reasonably assign a different label.

ABSENT — You cannot identify meaningful textual evidence for the assigned label. The label appears to over-interpret the text, project features onto neutral language, or mischaracterize the discursive content. Relative to other cases in this batch, the evidentiary grounding is weakest or nonexistent.

Draw the boundaries between tiers based on meaningful differences in evidentiary strength across the full batch. Do not default to lenient or strict standards — let the comparative range of this batch determine where the lines fall.

For each case, output:

- case_id: The case identifier.
- classification: CLEAR, BORDERLINE, or ABSENT.
- confidence: 1-5 score.
    1 = very uncertain, adjacent category equally plausible
    2 = somewhat uncertain, marginal case
    3 = moderately confident
    4 = confident
    5 = very confident, classification is obvious
- evidence: 2-3 sentences identifying the specific textual features you found (or failed to find) that support or undermine the assigned label. Do not speculate beyond what is in the text.
- strongest_feature: The single discourse feature most clearly present in the text, in your independent reading.
- weakest_feature: The discourse feature most weakly supported or not present. Write "none" if all are well-supported.

Output your response as a JSON array of objects, one per case, ordered by case_id. Do not include any text outside the JSON array.
```

---

## Evaluator user template

```
You are evaluating a batch of discourse analyses. The analyst was given the following instructions:

<analytical_framework>
{system_prompt}

The analyst was then asked to apply this framework to each text segment using the following task instruction:

{user_template}
</analytical_framework>

Below is a batch of cases. Each case contains the original text and the label assigned by the analyst. The analyst's reasoning is not included — form your own judgment.

<batch>
{batch_items}
</batch>

Each case is formatted as:

<case id="{case_id}">
<text>
{text}
</text>
<assigned_label>
{label}
</assigned_label>
</case>

First review all cases to calibrate your standards, then evaluate each case. Respond with a JSON array only.
```

---

## Pipeline implementation: evaluate_filter.py

### Core logic

```python
import polars as pl
import json
import random
import ollama
from pathlib import Path


def load_prompts(system_prompt_path: str, user_template_path: str) -> tuple[str, str]:
    """Load the original analyst prompts so they can be passed to the evaluator."""
    system_prompt = Path(system_prompt_path).read_text()
    user_template = Path(user_template_path).read_text()
    return system_prompt, user_template


def build_batch_items(df_batch: pl.DataFrame) -> str:
    """Format a batch of cases into the XML structure expected by the evaluator."""
    items = []
    for row in df_batch.iter_rows(named=True):
        case_id = f"{row['source_file']}__p{row['paragraph_index']}"
        item = (
            f'<case id="{case_id}">\n'
            f'<text>\n{row["text"]}\n</text>\n'
            f'<assigned_label>\n{row["model_evaluation"]}\n</assigned_label>\n'
            f'</case>'
        )
        items.append(item)
    return "\n\n".join(items)


def build_evaluator_prompt(
    batch_items: str,
    analyst_system_prompt: str,
    analyst_user_template: str,
    evaluator_user_template: str,
) -> str:
    """Assemble the full evaluator prompt for one batch."""
    prompt = evaluator_user_template
    prompt = prompt.replace("{system_prompt}", analyst_system_prompt)
    prompt = prompt.replace("{user_template}", analyst_user_template)
    prompt = prompt.replace("{batch_items}", batch_items)
    return prompt


def parse_evaluator_response(content: str) -> list[dict]:
    """Parse JSON array from model response, with fallback extraction."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON array from surrounding text
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
        raise ValueError(f"Could not parse evaluator response as JSON: {content[:200]}...")


def estimate_batch_size(df: pl.DataFrame, max_context_tokens: int) -> int:
    """Estimate how many cases fit in one batch given model context limits."""
    sample = df.head(min(10, len(df)))
    total_tokens = 0
    for row in sample.iter_rows(named=True):
        text_tokens = len(row["text"].split()) * 1.3
        label_tokens = len(row["model_evaluation"].split()) * 1.3
        total_tokens += text_tokens + label_tokens
    avg_per_case = total_tokens / len(sample)

    # Reserve for: system prompt (~1000), analyst prompts in template (~1500),
    # output (~150 per case)
    available = max_context_tokens - 2500
    per_case_total = avg_per_case + 150
    batch_size = int(available / per_case_total)

    return max(10, min(50, batch_size))
```

### Batching with shuffle and overlap

```python
def create_batches(
    df: pl.DataFrame,
    batch_size: int,
    overlap: int = 5,
    shuffle: bool = True,
    seed: int = 42,
) -> list[dict]:
    """
    Create batches with optional shuffling and overlap anchors.

    Returns a list of batch dicts, each containing:
        - 'indices': row indices in the shuffled DataFrame
        - 'anchor_indices': indices of overlap/anchor cases (from previous batch)
        - 'new_indices': indices of cases unique to this batch
    """
    n = len(df)
    indices = list(range(n))

    if shuffle:
        random.seed(seed)
        random.shuffle(indices)

    batches = []
    i = 0
    prev_tail = []

    while i < n:
        # Anchor cases from previous batch
        anchors = prev_tail[-overlap:] if prev_tail and overlap > 0 else []

        # New cases for this batch
        remaining_slots = batch_size - len(anchors)
        new = indices[i : i + remaining_slots]
        i += remaining_slots

        batch_indices = anchors + new

        batches.append({
            "indices": batch_indices,
            "anchor_indices": set(anchors),
            "new_indices": set(new),
        })

        prev_tail = new  # tail of current batch becomes anchor source for next

    return batches
```

### Running evaluation with anchor consistency checks

```python
def run_evaluation(
    df: pl.DataFrame,
    batches: list[dict],
    evaluator_model: str,
    analyst_system_prompt: str,
    analyst_user_template: str,
    evaluator_user_template: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Run batch evaluation across all batches.

    Returns:
        - results_df: full evaluation results
        - anchor_report: consistency report for anchor cases
    """
    all_results = []
    anchor_classifications = {}  # case_id -> list of (batch_idx, classification)

    for batch_idx, batch in enumerate(batches):
        # Slice the DataFrame for this batch
        df_batch = df[batch["indices"]]

        # Build and send prompt
        batch_items = build_batch_items(df_batch)
        prompt = build_evaluator_prompt(
            batch_items,
            analyst_system_prompt,
            analyst_user_template,
            evaluator_user_template,
        )

        response = ollama.chat(
            model=evaluator_model,
            messages=[{"role": "user", "content": prompt}],
        )

        evaluations = parse_evaluator_response(response["message"]["content"])

        # Track anchor case consistency
        for ev in evaluations:
            case_id = ev["case_id"]
            row_idx = _case_id_to_index(case_id, df)

            if row_idx in batch["anchor_indices"]:
                if case_id not in anchor_classifications:
                    anchor_classifications[case_id] = []
                anchor_classifications[case_id].append({
                    "batch": batch_idx,
                    "classification": ev["classification"],
                    "confidence": ev["confidence"],
                })

            # Only store results for new (non-anchor) cases,
            # unless this is the first appearance of an anchor
            if row_idx in batch["new_indices"]:
                all_results.append(ev)
            elif row_idx in batch["anchor_indices"]:
                # Keep the first evaluation for anchors, flag if it changes
                existing = [r for r in all_results if r["case_id"] == case_id]
                if not existing:
                    all_results.append(ev)

    # Build anchor consistency report
    anchor_rows = []
    for case_id, entries in anchor_classifications.items():
        classifications = [e["classification"] for e in entries]
        consistent = len(set(classifications)) == 1
        anchor_rows.append({
            "case_id": case_id,
            "classifications": classifications,
            "consistent": consistent,
            "batches": [e["batch"] for e in entries],
        })

    results_df = pl.DataFrame(all_results)
    anchor_report = pl.DataFrame(anchor_rows) if anchor_rows else pl.DataFrame()

    return results_df, anchor_report
```

### Post-hoc calibration check

```python
def calibration_report(
    results_df: pl.DataFrame,
    batches: list[dict],
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Check whether the CLEAR/BORDERLINE/ABSENT distribution is stable across batches.

    Large deviations suggest the evaluator's threshold drifted between batches.
    """
    rows = []

    for batch_idx, batch in enumerate(batches):
        # Get case_ids for new (non-anchor) cases in this batch
        batch_case_ids = set()
        for idx in batch["new_indices"]:
            row = df.row(idx, named=True)
            case_id = f"{row['source_file']}__p{row['paragraph_index']}"
            batch_case_ids.add(case_id)

        # Filter results for this batch
        batch_results = results_df.filter(
            pl.col("case_id").is_in(batch_case_ids)
        )

        if len(batch_results) == 0:
            continue

        total = len(batch_results)
        clear_pct = len(batch_results.filter(pl.col("classification") == "CLEAR")) / total
        border_pct = len(batch_results.filter(pl.col("classification") == "BORDERLINE")) / total
        absent_pct = len(batch_results.filter(pl.col("classification") == "ABSENT")) / total
        avg_confidence = batch_results["confidence"].mean()

        rows.append({
            "batch": batch_idx,
            "n_cases": total,
            "pct_clear": round(clear_pct, 3),
            "pct_borderline": round(border_pct, 3),
            "pct_absent": round(absent_pct, 3),
            "avg_confidence": round(avg_confidence, 2),
        })

    report = pl.DataFrame(rows)

    # Flag batches that deviate significantly from the overall distribution
    if len(report) > 1:
        overall_clear = report["pct_clear"].mean()
        overall_border = report["pct_borderline"].mean()

        report = report.with_columns([
            (pl.col("pct_clear") - overall_clear).abs().alias("clear_deviation"),
            (pl.col("pct_borderline") - overall_border).abs().alias("borderline_deviation"),
        ])

        # Flag if deviation exceeds 20 percentage points from the mean
        report = report.with_columns(
            ((pl.col("clear_deviation") > 0.20) | (pl.col("borderline_deviation") > 0.20))
            .alias("drift_flag")
        )

    return report
```

### Main entry point

```python
def evaluate_and_filter(
    evaluations_path: str,
    analyst_system_prompt_path: str,
    analyst_user_template_path: str,
    evaluator_user_template_path: str,
    evaluator_model: str,
    output_dir: str,
    batch_size: int | None = None,
    context_window: int = 32000,
    overlap: int = 5,
    shuffle: bool = True,
    threshold: str = "clear",      # "clear" or "borderline"
    min_confidence: int = 1,
    seed: int = 42,
):
    """Full evaluation pipeline: batch, evaluate, check consistency, filter."""

    # Load data
    df = pl.read_parquet(evaluations_path)
    analyst_sys, analyst_user = load_prompts(
        analyst_system_prompt_path, analyst_user_template_path
    )
    eval_template = Path(evaluator_user_template_path).read_text()

    # Determine batch size
    if batch_size is None:
        batch_size = estimate_batch_size(df, context_window)
    print(f"Using batch size: {batch_size}")

    # Create batches
    batches = create_batches(df, batch_size, overlap=overlap, shuffle=shuffle, seed=seed)
    print(f"Created {len(batches)} batches")

    # Run evaluation
    results_df, anchor_report = run_evaluation(
        df, batches, evaluator_model,
        analyst_sys, analyst_user, eval_template,
    )

    # Post-hoc calibration
    cal_report = calibration_report(results_df, batches, df)

    # Print diagnostics
    print("\n--- Calibration Report ---")
    print(cal_report)

    if "drift_flag" in cal_report.columns:
        drifted = cal_report.filter(pl.col("drift_flag"))
        if len(drifted) > 0:
            print(f"\nWARNING: {len(drifted)} batch(es) show classification drift.")
            print("Consider re-running with different batch composition or larger batches.")

    if len(anchor_report) > 0:
        inconsistent = anchor_report.filter(~pl.col("consistent"))
        print(f"\n--- Anchor Consistency ---")
        print(f"Anchor cases: {len(anchor_report)}")
        print(f"Inconsistent: {len(inconsistent)}")
        if len(inconsistent) > 0:
            print("Inconsistent anchors:")
            print(inconsistent)

    # Join results back to original DataFrame
    full_df = df.join(
        results_df.select(["case_id", "classification", "confidence",
                           "evidence", "strongest_feature", "weakest_feature"]),
        left_on=pl.concat_str([
            pl.col("source_file"),
            pl.lit("__p"),
            pl.col("paragraph_index").cast(pl.Utf8),
        ]),
        right_on="case_id",
        how="left",
    )

    # Save full results
    out = Path(output_dir)
    full_df.write_parquet(out / "evaluated_full.parquet")

    # Apply filter
    allowed = ["CLEAR"] if threshold == "clear" else ["CLEAR", "BORDERLINE"]
    filtered = full_df.filter(
        pl.col("classification").is_in(allowed)
        & (pl.col("confidence") >= min_confidence)
    )
    filtered.write_parquet(out / "evaluated_filtered.parquet")

    # Save diagnostics
    cal_report.write_parquet(out / "calibration_report.parquet")
    if len(anchor_report) > 0:
        anchor_report.write_parquet(out / "anchor_consistency.parquet")

    print(f"\nFull results: {len(full_df)} cases")
    print(f"Filtered results: {len(filtered)} cases ({len(filtered)/len(full_df)*100:.1f}%)")

    return full_df, filtered
```

### CLI integration

```bash
python cli.py filter \
  --evaluations data/evaluations/paper.parquet \
  --analyst-system-prompt prompts/discourse_analysis_system_prompt.txt \
  --analyst-user-template prompts/discourse_user_template.txt \
  --evaluator-model llama3-evaluator \
  --batch-size auto \
  --context-window 32000 \
  --shuffle \
  --overlap 5 \
  --threshold clear \
  --min-confidence 4 \
  --seed 42 \
  --output data/evaluations/
```

### Output files

| File | Contents |
|------|----------|
| `evaluated_full.parquet` | All cases with evaluator columns appended |
| `evaluated_filtered.parquet` | Only cases passing threshold + confidence filter |
| `calibration_report.parquet` | Per-batch distribution stats with drift flags |
| `anchor_consistency.parquet` | Anchor cases with cross-batch classification comparison |
