"""
build_stimuli.py

Takes the raw extraction (output of extract_separable_verbs.py) and produces a
curated, distance-balanced stimulus set for the scoring experiment.

Steps:
  1. Filter out noisy rows (non-alphabetic prefixes/verbs, single-letter junk).
  2. Restrict to a chosen list of high-frequency verb stems that have several
     distinct, meaning-changing prefixes attested (good minimal-pair material).
  3. Bucket each instance by distance (short / medium / long — thresholds configurable).
  4. Sample a balanced number of items per (stem, bucket) cell.
  5. For each sampled item, attach distractor prefixes: other prefixes attested
     with the same verb stem elsewhere in the corpus, excluding the correct one.

Usage:
    python src/build_stimuli.py --input data/extracted/separable_verbs_extracted.csv \
        --output data/stimuli/stimulus_set_v1.csv --n-per-cell 3 --seed 42

    # to see stem/bucket coverage without writing a file, e.g. before deciding on stems:
    python src/build_stimuli.py --input data/extracted/separable_verbs_extracted.csv --report-only
"""

import argparse
import csv
import random
import sys
from collections import defaultdict

# Default candidate stems: common, high-frequency verbs whose prefixes give
# clearly DIFFERENT meanings (not near-synonyms), which makes for stronger,
# less ambiguous minimal-pair items. Edit this list freely.
DEFAULT_STEMS = [
    "kommen", "gehen", "stellen", "geben", "nehmen", "setzen",
    "ziehen", "machen", "fallen", "laufen", "bringen", "schlagen",
    "führen", "treten", "finden", "halten", "sehen",
]


def is_clean(row):
    p, v = row["prefix_lemma"], row["verb_lemma"]
    return p.isalpha() and v.isalpha() and len(p) >= 2


def bucket(distance, short_max, medium_max):
    d = int(distance)
    if d <= short_max:
        return "short"
    elif d <= medium_max:
        return "medium"
    else:
        return "long"


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def report_coverage(rows, stems, short_max, medium_max):
    stem_to_prefixes = defaultdict(set)
    for r in rows:
        stem_to_prefixes[r["verb_lemma"]].add(r["prefix_lemma"])

    grouped = defaultdict(list)
    for r in rows:
        if r["verb_lemma"] in stems:
            grouped[(r["verb_lemma"], bucket(r["distance"], short_max, medium_max))].append(r)

    print(f"{'stem':<12} {'#prefixes':<10} {'short':<8} {'medium':<8} {'long':<8}")
    for stem in stems:
        n_prefixes = len(stem_to_prefixes.get(stem, []))
        counts = {b: len(grouped[(stem, b)]) for b in ["short", "medium", "long"]}
        print(f"{stem:<12} {n_prefixes:<10} {counts['short']:<8} {counts['medium']:<8} {counts['long']:<8}")


def build(rows, stems, n_per_cell, short_max, medium_max, n_distractors, seed):
    random.seed(seed)

    stem_to_prefixes = defaultdict(set)
    for r in rows:
        stem_to_prefixes[r["verb_lemma"]].add(r["prefix_lemma"])

    grouped = defaultdict(list)
    for r in rows:
        if r["verb_lemma"] in stems:
            grouped[(r["verb_lemma"], bucket(r["distance"], short_max, medium_max))].append(r)

    items = []
    item_id = 1
    for stem in stems:
        prefixes_for_stem = stem_to_prefixes.get(stem, set())
        for b in ["short", "medium", "long"]:
            pool = grouped[(stem, b)]
            if not pool:
                continue
            sample = random.sample(pool, min(n_per_cell, len(pool)))
            for r in sample:
                correct_prefix = r["prefix_lemma"]
                distractor_pool = [p for p in prefixes_for_stem if p != correct_prefix]
                if not distractor_pool:
                    continue
                distractors = random.sample(
                    distractor_pool, min(n_distractors, len(distractor_pool))
                )
                items.append({
                    "item_id": item_id,
                    "verb_stem": stem,
                    "distance_bucket": b,
                    "n_intervening_tokens": r["distance"],
                    "context_sentence": r["context_up_to_prefix"],
                    "correct_prefix": correct_prefix,
                    "distractor_prefixes": ";".join(distractors),
                    "full_original_sentence": r["text"],
                    "intervening_text": r["intervening_text"],
                    "sent_id": r["sent_id"],
                    "source_file": r["source_file"],
                })
                item_id += 1

    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to extracted CSV.")
    parser.add_argument("--output", default="stimulus_set_v1.csv")
    parser.add_argument("--stems", nargs="*", default=None,
                         help="Verb stems to include (default: a built-in curated list).")
    parser.add_argument("--n-per-cell", type=int, default=5,
                         help="Number of items to sample per (stem, distance bucket).")
    parser.add_argument("--n-distractors", type=int, default=2,
                         help="Number of distractor prefixes per item.")
    parser.add_argument("--short-max", type=int, default=2,
                         help="Max intervening-token count still counted as 'short'.")
    parser.add_argument("--medium-max", type=int, default=7,
                         help="Max intervening-token count still counted as 'medium' "
                              "(anything above is 'long').")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-only", action="store_true",
                         help="Just print stem/bucket coverage counts and exit, "
                              "without writing a stimulus file. Useful for deciding "
                              "on --stems and --n-per-cell before committing.")
    args = parser.parse_args()

    rows = load_rows(args.input)
    rows = [r for r in rows if is_clean(r)]
    print(f"Loaded {len(rows)} clean rows from {args.input}", file=sys.stderr)

    stems = args.stems if args.stems else DEFAULT_STEMS

    if args.report_only:
        report_coverage(rows, stems, args.short_max, args.medium_max)
        return

    items = build(rows, stems, args.n_per_cell, args.short_max, args.medium_max,
                   args.n_distractors, args.seed)

    if not items:
        print("No items generated — check that --stems match verb_lemma values "
              "in the input file.", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(items[0].keys()))
        writer.writeheader()
        writer.writerows(items)

    from collections import Counter
    print(f"\nGenerated {len(items)} stimulus items -> {args.output}", file=sys.stderr)
    print(f"Per stem: {dict(Counter(i['verb_stem'] for i in items))}", file=sys.stderr)
    print(f"Per bucket: {dict(Counter(i['distance_bucket'] for i in items))}", file=sys.stderr)


if __name__ == "__main__":
    main()