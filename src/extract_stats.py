import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(values, n_boot=5000, seed=42):
    """95% bootstrap CI for a binary accuracy vector."""
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return [None, None]

    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)

    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def safe_corr(x, y):
    """Pearson correlation, returning None when undefined."""
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    mask = x.notna() & y.notna()

    if mask.sum() < 3:
        return None

    if x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return None

    return float(x[mask].corr(y[mask]))


def clean_value(value):
    """Convert numpy values into JSON-safe Python values."""
    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    return value


def main():
    parser = argparse.ArgumentParser(
        description="Generate statistics from separable-verb prefix scoring CSV."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input scoring CSV"
    )

    parser.add_argument(
        "--output-dir",
        default="results/statistics",
        help="Directory where reports are saved"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} rows from {input_path}")

    # ------------------------------------------------------------
    # Clean / normalize columns
    # ------------------------------------------------------------

    required_columns = [
        "item_id",
        "verb_stem",
        "distance_bucket",
        "n_intervening_tokens",
        "context_sentence",
        "correct_prefix",
        "distractor_prefixes",
        "correct_logprob",
        "correct_logprob_per_token",
        "best_distractor_logprob",
        "margin",
        "model_prefers_correct",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Convert numerical columns
    numeric_columns = [
        "n_intervening_tokens",
        "correct_logprob",
        "correct_logprob_per_token",
        "best_distractor_logprob",
        "margin",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize boolean column
    if df["model_prefers_correct"].dtype == object:
        df["model_prefers_correct"] = (
            df["model_prefers_correct"]
            .astype(str)
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            })
        )

    # ------------------------------------------------------------
    # Basic dataset statistics
    # ------------------------------------------------------------

    total_items = len(df)
    valid_predictions = df["model_prefers_correct"].notna().sum()

    correct = int(
        df["model_prefers_correct"]
        .fillna(False)
        .sum()
    )

    accuracy = (
        correct / valid_predictions
        if valid_predictions > 0
        else None
    )

    accuracy_ci = bootstrap_ci(
        df.loc[
            df["model_prefers_correct"].notna(),
            "model_prefers_correct"
        ].astype(int)
    )

    # ------------------------------------------------------------
    # Distance statistics
    # ------------------------------------------------------------

    distance_stats = (
        df.groupby("distance_bucket", dropna=False)
        .agg(
            n_items=("item_id", "count"),
            accuracy=("model_prefers_correct", "mean"),
            mean_distance=("n_intervening_tokens", "mean"),
            median_distance=("n_intervening_tokens", "median"),
            mean_margin=("margin", "mean"),
            median_margin=("margin", "median"),
            mean_correct_logprob=("correct_logprob", "mean"),
            mean_correct_logprob_per_token=(
                "correct_logprob_per_token",
                "mean",
            ),
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # Exact-distance statistics
    # ------------------------------------------------------------

    exact_distance_stats = (
        df.groupby("n_intervening_tokens", dropna=False)
        .agg(
            n_items=("item_id", "count"),
            accuracy=("model_prefers_correct", "mean"),
            mean_margin=("margin", "mean"),
            median_margin=("margin", "median"),
        )
        .reset_index()
        .sort_values("n_intervening_tokens")
    )

    # ------------------------------------------------------------
    # Verb-stem statistics
    # ------------------------------------------------------------

    stem_stats = (
        df.groupby("verb_stem", dropna=False)
        .agg(
            n_items=("item_id", "count"),
            accuracy=("model_prefers_correct", "mean"),
            mean_distance=("n_intervening_tokens", "mean"),
            mean_margin=("margin", "mean"),
            median_margin=("margin", "median"),
            mean_correct_logprob=(
                "correct_logprob",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("accuracy")
    )

    # ------------------------------------------------------------
    # Candidate statistics
    # ------------------------------------------------------------

    def count_distractors(value):
        if pd.isna(value) or str(value).strip() == "":
            return 0

        return len([
            x for x in str(value).split(",")
            if x.strip()
        ])

    df["n_distractors"] = df["distractor_prefixes"].apply(
        count_distractors
    )

    candidate_stats = (
        df.groupby("n_distractors")
        .agg(
            n_items=("item_id", "count"),
            accuracy=("model_prefers_correct", "mean"),
            mean_margin=("margin", "mean"),
        )
        .reset_index()
        .sort_values("n_distractors")
    )

    # ------------------------------------------------------------
    # Correlations
    # ------------------------------------------------------------

    distance_margin_corr = safe_corr(
        df["n_intervening_tokens"],
        df["margin"],
    )

    distance_accuracy_corr = safe_corr(
        df["n_intervening_tokens"],
        df["model_prefers_correct"].astype(float),
    )

    # ------------------------------------------------------------
    # Margin statistics
    # ------------------------------------------------------------

    margin_stats = {
        "mean": clean_value(df["margin"].mean()),
        "median": clean_value(df["margin"].median()),
        "std": clean_value(df["margin"].std()),
        "min": clean_value(df["margin"].min()),
        "max": clean_value(df["margin"].max()),
        "positive_margin_rate": clean_value(
            (df["margin"] > 0).mean()
        ),
        "negative_margin_rate": clean_value(
            (df["margin"] < 0).mean()
        ),
        "zero_margin_rate": clean_value(
            (df["margin"] == 0).mean()
        ),
    }

    # ------------------------------------------------------------
    # Data quality
    # ------------------------------------------------------------

    data_quality = {
        "rows": int(len(df)),
        "unique_items": int(df["item_id"].nunique()),
        "duplicate_item_ids": int(
            df["item_id"].duplicated().sum()
        ),
        "missing_correct_logprob": int(
            df["correct_logprob"].isna().sum()
        ),
        "missing_margin": int(
            df["margin"].isna().sum()
        ),
        "missing_distance": int(
            df["n_intervening_tokens"].isna().sum()
        ),
        "missing_prediction": int(
            df["model_prefers_correct"].isna().sum()
        ),
    }

    # ------------------------------------------------------------
    # Model name
    # ------------------------------------------------------------

    model_name = input_path.stem

    # ------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------

    report = {
        "input_file": str(input_path),
        "model": model_name,

        "overall": {
            "total_items": total_items,
            "valid_predictions": int(valid_predictions),
            "correct_predictions": correct,
            "accuracy": accuracy,
            "accuracy_95_ci": accuracy_ci,
        },

        "margin": margin_stats,

        "correlations": {
            "distance_vs_margin_pearson_r": distance_margin_corr,
            "distance_vs_accuracy_pearson_r": distance_accuracy_corr,
        },

        "distance_buckets": (
            distance_stats
            .replace({np.nan: None})
            .to_dict(orient="records")
        ),

        "exact_distance": (
            exact_distance_stats
            .replace({np.nan: None})
            .to_dict(orient="records")
        ),

        "verb_stems": (
            stem_stats
            .replace({np.nan: None})
            .to_dict(orient="records")
        ),

        "candidate_counts": (
            candidate_stats
            .replace({np.nan: None})
            .to_dict(orient="records")
        ),

        "data_quality": data_quality,
    }

    json_path = output_dir / f"{model_name}_statistics.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------
    # CSV summary tables
    # ------------------------------------------------------------

    distance_stats.to_csv(
        output_dir / f"{model_name}_by_distance_bucket.csv",
        index=False,
    )

    exact_distance_stats.to_csv(
        output_dir / f"{model_name}_by_exact_distance.csv",
        index=False,
    )

    stem_stats.to_csv(
        output_dir / f"{model_name}_by_stem.csv",
        index=False,
    )

    candidate_stats.to_csv(
        output_dir / f"{model_name}_by_candidate_count.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Human-readable TXT report
    # ------------------------------------------------------------

    txt_path = output_dir / f"{model_name}_statistics.txt"

    with open(txt_path, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write("SEPARABLE VERB PREFIX SCORING STATISTICS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Input: {input_path}\n")
        f.write(f"Model: {model_name}\n\n")

        f.write("OVERALL\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total items:        {total_items}\n")
        f.write(f"Valid predictions:  {valid_predictions}\n")
        f.write(f"Correct:            {correct}\n")

        if accuracy is not None:
            f.write(f"Accuracy:           {accuracy:.4f}\n")
            f.write(
                f"95% bootstrap CI:   "
                f"[{accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f}]\n"
            )

        f.write("\n")

        f.write("MARGIN\n")
        f.write("-" * 70 + "\n")
        for key, value in margin_stats.items():
            if value is not None:
                f.write(f"{key}: {value:.4f}\n")

        f.write("\n")

        f.write("CORRELATIONS\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Distance vs margin (Pearson r): "
            f"{distance_margin_corr}\n"
        )
        f.write(
            "Distance vs accuracy (Pearson r): "
            f"{distance_accuracy_corr}\n"
        )

        f.write("\n")

        f.write("ACCURACY BY DISTANCE BUCKET\n")
        f.write("-" * 70 + "\n")
        f.write(
            distance_stats.to_string(index=False)
        )
        f.write("\n\n")

        f.write("ACCURACY BY EXACT DISTANCE\n")
        f.write("-" * 70 + "\n")
        f.write(
            exact_distance_stats.to_string(index=False)
        )
        f.write("\n\n")

        f.write("ACCURACY BY VERB STEM\n")
        f.write("-" * 70 + "\n")
        f.write(
            stem_stats.to_string(index=False)
        )
        f.write("\n\n")

        f.write("ACCURACY BY NUMBER OF DISTRACTORS\n")
        f.write("-" * 70 + "\n")
        f.write(
            candidate_stats.to_string(index=False)
        )
        f.write("\n\n")

        f.write("DATA QUALITY\n")
        f.write("-" * 70 + "\n")

        for key, value in data_quality.items():
            f.write(f"{key}: {value}\n")

    print()
    print("=" * 70)
    print("STATISTICS GENERATED")
    print("=" * 70)
    print(f"JSON: {json_path}")
    print(f"TXT:  {txt_path}")
    print()
    print(
        f"Accuracy: {accuracy:.4f}"
        if accuracy is not None
        else "Accuracy: N/A"
    )
    print(f"Mean margin: {df['margin'].mean():.4f}")
    print()
    print("Accuracy by distance:")
    print(
        distance_stats[
            [
                "distance_bucket",
                "n_items",
                "accuracy",
                "mean_margin",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()