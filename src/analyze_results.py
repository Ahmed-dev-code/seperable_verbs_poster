"""
analyze_results.py

Takes one or more results_<model>.csv files (output of score_prefixes.py) and
produces the core poster figures:
  1. Accuracy vs. distance bucket, one line per model
  2. Margin (correct - best distractor logprob) vs. distance bucket, one line per model
  3. Per-stem breakdown (accuracy per verb stem, averaged across distance)
  4. A summary table (printed + saved as CSV)

Usage:
    python analyze_results.py --results results/results_german-gpt2.csv results/results_gerpt2-large.csv
    python analyze_results.py --glob "results/results_*.csv"
"""

import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def model_name_from_path(path):
    """Derive a short model label from the filename, e.g. results_german-gpt2.csv -> german-gpt2"""
    stem = Path(path).stem
    if stem.startswith("results_"):
        stem = stem[len("results_"):]
    return stem


def load_results(paths):
    """Load all results files, tagging each row with its model name."""
    all_rows = []
    for path in paths:
        model = model_name_from_path(path)
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["model"] = model
                all_rows.append(row)
    return all_rows


BUCKET_ORDER = ["short", "medium", "long"]


def accuracy_by_bucket(rows):
    """{ model: { bucket: (accuracy, n_items) } }"""
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [n_correct, n_total]
    for r in rows:
        model = r["model"]
        bucket = r["distance_bucket"]
        counts[model][bucket][1] += 1
        counts[model][bucket][0] += int(r["model_prefers_correct"])

    result = {}
    for model, buckets in counts.items():
        result[model] = {}
        for b in BUCKET_ORDER:
            if b in buckets:
                n_correct, n_total = buckets[b]
                result[model][b] = (n_correct / n_total, n_total)
    return result


def margin_by_bucket(rows):
    """{ model: { bucket: mean_margin } }"""
    sums = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            margin = float(r["margin"])
        except (ValueError, KeyError):
            continue
        sums[r["model"]][r["distance_bucket"]].append(margin)

    result = {}
    for model, buckets in sums.items():
        result[model] = {}
        for b in BUCKET_ORDER:
            if b in buckets and buckets[b]:
                result[model][b] = sum(buckets[b]) / len(buckets[b])
    return result


def accuracy_by_stem(rows):
    """{ model: { stem: accuracy } }"""
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        model = r["model"]
        stem = r.get("verb_stem", "unknown")
        counts[model][stem][1] += 1
        counts[model][stem][0] += int(r["model_prefers_correct"])

    result = {}
    for model, stems in counts.items():
        result[model] = {}
        for stem, (n_correct, n_total) in stems.items():
            result[model][stem] = n_correct / n_total if n_total else float("nan")
    return result


def plot_accuracy_vs_distance(acc_by_bucket, out_path):
    plt.figure(figsize=(7, 5))
    for model, buckets in acc_by_bucket.items():
        xs = [b for b in BUCKET_ORDER if b in buckets]
        ys = [buckets[b][0] for b in xs]
        plt.plot(xs, ys, marker="o", label=model)
    plt.xlabel("Distance between verb stem and prefix")
    plt.ylabel("Accuracy (model prefers correct prefix)")
    plt.title("Separable-prefix resolution accuracy vs. distance")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_margin_vs_distance(margin_by_bucket_data, out_path):
    plt.figure(figsize=(7, 5))
    for model, buckets in margin_by_bucket_data.items():
        xs = [b for b in BUCKET_ORDER if b in buckets]
        ys = [buckets[b] for b in xs]
        plt.plot(xs, ys, marker="o", label=model)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Distance between verb stem and prefix")
    plt.ylabel("Mean margin (log P(correct) - log P(best distractor))")
    plt.title("Model confidence margin vs. distance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_accuracy_by_stem(acc_by_stem, out_path):
    models = list(acc_by_stem.keys())
    all_stems = sorted({s for m in acc_by_stem.values() for s in m.keys()})

    fig, ax = plt.subplots(figsize=(10, 5))
    n_models = len(models)
    bar_width = 0.8 / max(n_models, 1)

    for i, model in enumerate(models):
        ys = [acc_by_stem[model].get(s, 0) for s in all_stems]
        xs = [j + i * bar_width for j in range(len(all_stems))]
        ax.bar(xs, ys, width=bar_width, label=model)

    ax.set_xticks([j + (n_models - 1) * bar_width / 2 for j in range(len(all_stems))])
    ax.set_xticklabels(all_stems, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by verb stem")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def print_and_save_summary(acc_by_bucket, out_path):
    fieldnames = ["model"] + BUCKET_ORDER + ["overall"]
    rows_out = []
    print(f"\n{'model':<20} " + " ".join(f"{b:>10}" for b in BUCKET_ORDER) + f" {'overall':>10}")
    for model, buckets in acc_by_bucket.items():
        total_correct, total_n = 0, 0
        row = {"model": model}
        line = f"{model:<20} "
        for b in BUCKET_ORDER:
            if b in buckets:
                acc, n = buckets[b]
                total_correct += acc * n
                total_n += n
                line += f"{acc:>10.1%} "
                row[b] = f"{acc:.3f}"
            else:
                line += f"{'--':>10} "
                row[b] = ""
        overall = total_correct / total_n if total_n else float("nan")
        line += f"{overall:>10.1%}"
        row["overall"] = f"{overall:.3f}"
        print(line)
        rows_out.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nSaved summary table to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="*", default=[],
                         help="One or more results CSV paths.")
    parser.add_argument("--glob", action="append", default=[],
                         help="Glob pattern(s) to expand manually.")
    parser.add_argument("--outdir", default="results/figures",
                         help="Directory to save figures and summary CSV into.")
    args = parser.parse_args()

    paths = list(args.results)
    for pattern in args.glob:
        paths.extend(glob.glob(pattern, recursive=True))
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]

    if not paths:
        parser.error("No results files given. Use --results or --glob.")

    print(f"Loading {len(paths)} results file(s):", file=sys.stderr)
    for p in paths:
        print(f"  {p} -> model = '{model_name_from_path(p)}'", file=sys.stderr)

    rows = load_results(paths)
    print(f"Total scored items loaded: {len(rows)}", file=sys.stderr)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    acc_bucket = accuracy_by_bucket(rows)
    marg_bucket = margin_by_bucket(rows)
    acc_stem = accuracy_by_stem(rows)

    plot_accuracy_vs_distance(acc_bucket, f"{args.outdir}/accuracy_vs_distance.png")
    plot_margin_vs_distance(marg_bucket, f"{args.outdir}/margin_vs_distance.png")
    plot_accuracy_by_stem(acc_stem, f"{args.outdir}/accuracy_by_stem.png")
    print_and_save_summary(acc_bucket, f"{args.outdir}/summary_accuracy.csv")


if __name__ == "__main__":
    main()