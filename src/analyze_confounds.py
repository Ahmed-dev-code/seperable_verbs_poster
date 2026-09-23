"""
analyze_confounds.py

Checks two potential confounds in the full-ranking results, using only the
standard library (no numpy/scipy) to avoid any risk of compiled-extension
issues like the spaCy one you hit.

  1. CANDIDATE-SET SIZE vs. ACCURACY (per verb stem)
     Some stems (e.g. "kommen") have far more attested prefixes than others
     (e.g. "finden"), so they compete against more candidates in the
     full-ranking setup. If stems with more candidates simply score lower
     regardless of distance, that's a confound worth naming explicitly.

  2. CONTEXT LENGTH vs. CORRECTNESS, WITHIN each distance bucket
     Distance and context length are correlated by construction (more
     intervening tokens = more preceding words), so a raw overall
     correlation would just reproduce the distance effect. This checks
     whether context length predicts correctness *within* a bucket (i.e.
     among items that are already all "medium" distance, does more context
     still help?) -- a stronger, more specific test of the "more context
     helps disambiguate" explanation.

Usage:
    python analyze_confounds.py --glob "results/results_fullrank_*.csv" \
        --outdir results/confounds
"""

import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path


def model_name_from_path(path):
    stem = Path(path).stem
    if stem.startswith("results_"):
        stem = stem[len("results_"):]
    return stem


def load_results(paths):
    all_rows = []
    for path in paths:
        model = model_name_from_path(path)
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["model"] = model
                all_rows.append(row)
    return all_rows


def pearson_r(xs, ys):
    """Pure-Python Pearson correlation coefficient. Returns None if undefined
    (e.g. fewer than 2 points or zero variance in either variable)."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x ** 0.5 * var_y ** 0.5)


# ---------------------------------------------------------------------------
# Confound 1: candidate-set size vs. per-stem accuracy
# ---------------------------------------------------------------------------

def candidate_size_vs_accuracy(rows):
    """
    For each model, computes per-stem (n_candidates, accuracy) pairs and the
    Pearson correlation between them across stems.
    Returns: { model: {"per_stem": {stem: (n_candidates, accuracy, n_items)},
                        "correlation": r_or_None} }
    """
    result = {}
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    for model, model_rows in by_model.items():
        stem_correct = defaultdict(lambda: [0, 0])  # stem -> [n_correct, n_total]
        stem_ncand = {}
        for r in model_rows:
            stem = r.get("verb_stem", "unknown")
            stem_correct[stem][1] += 1
            stem_correct[stem][0] += int(r["model_prefers_correct"])
            try:
                stem_ncand[stem] = int(r["n_candidates"])
            except (ValueError, KeyError):
                pass

        per_stem = {}
        xs, ys = [], []
        for stem, (n_correct, n_total) in stem_correct.items():
            acc = n_correct / n_total if n_total else float("nan")
            ncand = stem_ncand.get(stem)
            per_stem[stem] = (ncand, acc, n_total)
            if ncand is not None and acc == acc:
                xs.append(ncand)
                ys.append(acc)

        r_value = pearson_r(xs, ys)
        result[model] = {"per_stem": per_stem, "correlation": r_value}

    return result


# ---------------------------------------------------------------------------
# Confound 2: context length vs. correctness, WITHIN each distance bucket
# ---------------------------------------------------------------------------

def context_length_vs_correctness_within_bucket(rows):
    """
    For each model and each distance bucket, computes the Pearson correlation
    between context word count and correctness (0/1), among items already
    in that bucket -- i.e. controlling for distance bucket.
    Returns: { model: { bucket: (r_or_None, n_items) } }
    """
    result = {}
    by_model_bucket = defaultdict(lambda: defaultdict(list))
    for r in rows:
        model = r["model"]
        bucket = r.get("distance_bucket", "unknown")
        context = r.get("context_sentence", "")
        n_words = len(context.split())
        correct = int(r["model_prefers_correct"])
        by_model_bucket[model][bucket].append((n_words, correct))

    for model, buckets in by_model_bucket.items():
        result[model] = {}
        for bucket, pairs in buckets.items():
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            r_value = pearson_r(xs, ys)
            result[model][bucket] = (r_value, len(pairs))

    return result


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="*", default=[])
    parser.add_argument("--glob", action="append", default=[])
    parser.add_argument("--outdir", default="results/confounds")
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
    print(f"Total scored items loaded: {len(rows)}\n", file=sys.stderr)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # --- Confound 1 ---
    print("=" * 70)
    print("CONFOUND 1: candidate-set size (n_candidates) vs. per-stem accuracy")
    print("=" * 70)
    c1 = candidate_size_vs_accuracy(rows)
    c1_out_rows = []
    for model, data in c1.items():
        r_value = data["correlation"]
        r_str = f"{r_value:.3f}" if r_value is not None else "N/A"
        print(f"\nModel: {model}   Pearson r(n_candidates, accuracy) across stems = {r_str}")
        if r_value is not None:
            if r_value < -0.3:
                print("  -> negative correlation: stems with MORE candidates tend to score LOWER.")
                print("     This is a real confound -- some of the per-stem accuracy variance")
                print("     (and possibly some of the distance effect, if bucket composition")
                print("     differs by stem) may be driven by candidate-set size, not distance.")
            elif r_value > 0.3:
                print("  -> positive correlation (unexpected direction) -- worth double-checking.")
            else:
                print("  -> weak/no correlation: candidate-set size does not appear to be a major confound.")
        print(f"  {'stem':<12} {'n_candidates':<14} {'accuracy':<10} {'n_items':<8}")
        for stem, (ncand, acc, n) in sorted(data["per_stem"].items(),
                                              key=lambda x: (x[1][0] is None, x[1][0] or 0)):
            ncand_str = str(ncand) if ncand is not None else "?"
            print(f"  {stem:<12} {ncand_str:<14} {acc:<10.3f} {n:<8}")
            c1_out_rows.append({"model": model, "stem": stem, "n_candidates": ncand_str,
                                 "accuracy": f"{acc:.3f}", "n_items": n})

    with open(f"{args.outdir}/confound_candidate_size.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "stem", "n_candidates", "accuracy", "n_items"])
        writer.writeheader()
        writer.writerows(c1_out_rows)
    print(f"\nSaved to {args.outdir}/confound_candidate_size.csv")

    # --- Confound 2 ---
    print("\n" + "=" * 70)
    print("CONFOUND 2: context length vs. correctness, WITHIN each distance bucket")
    print("=" * 70)
    c2 = context_length_vs_correctness_within_bucket(rows)
    c2_out_rows = []
    for model, buckets in c2.items():
        print(f"\nModel: {model}")
        for bucket in ["short", "medium", "long"]:
            if bucket not in buckets:
                continue
            r_value, n = buckets[bucket]
            r_str = f"{r_value:.3f}" if r_value is not None else "N/A"
            print(f"  bucket={bucket:<8} n_items={n:<5} "
                  f"Pearson r(context_word_count, correctness) = {r_str}")
            c2_out_rows.append({"model": model, "bucket": bucket, "n_items": n,
                                 "correlation": r_str})

    with open(f"{args.outdir}/confound_context_length.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "bucket", "n_items", "correlation"])
        writer.writeheader()
        writer.writerows(c2_out_rows)
    print(f"\nSaved to {args.outdir}/confound_context_length.csv")

    print("\nInterpretation guide:")
    print("  r close to 0        -> no evidence context length matters beyond distance bucket")
    print("  r > ~0.2 (positive) -> longer context WITHIN the same bucket predicts higher accuracy")
    print("                         (supports the 'more disambiguating context' explanation)")
    print("  r < ~-0.2 (negative)-> longer context WITHIN the same bucket predicts LOWER accuracy")
    print("                         (would argue against that explanation)")


if __name__ == "__main__":
    main()