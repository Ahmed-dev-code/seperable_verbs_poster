"""
compute_distractor_difficulty.py

For every item in a stimulus CSV (output of build_stimuli.py), measures how
semantically CLOSE each distractor prefix is to the correct prefix, by
comparing the reconstructed verb meanings (e.g. "ankommen" vs "auskommen")
using pretrained word vectors. This lets you split items into "hard"
(close-meaning distractor, genuinely confusable) vs. "easy" (far-meaning
distractor, trivially wrong) and re-run the accuracy/distance analysis
separately for each — the current flat/ceiling result may hide a real
distance effect that only shows up once distractor difficulty is controlled.

No training, no API calls, fully local and free. Two backends:

  1. spaCy (recommended, easiest to set up):
       pip install spacy
       python -m spacy download de_core_news_lg
     de_core_news_lg ships with real word vectors (de_core_news_sm/md do NOT
     have usable vectors, so use lg).

  2. gensim + a local vector file (e.g. a downloaded fastText .vec/.bin file),
     if you'd rather not pull a spaCy model:
       pip install gensim
       python compute_distractor_difficulty.py --backend gensim \
           --vectors path/to/cc.de.300.vec ...

Usage:
    python compute_distractor_difficulty.py --backend spacy \
        --input data/stimuli/stimulus_set_v1.csv \
        --output data/stimuli/stimulus_set_v1_with_difficulty.csv
"""

import argparse
import csv
import sys


def reconstruct_verb(prefix, stem):
    """Naive reconstruction: prefix + stem, e.g. 'an' + 'kommen' -> 'ankommen'.
    This is the standard separable-verb formation rule and matches the
    'reconstructed_lemma' field already produced by extract_separable_verbs.py."""
    return prefix + stem


def cosine_sim(vec_a, vec_b):
    import numpy as np
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# spaCy backend
# ---------------------------------------------------------------------------

def get_vector_spacy(nlp, word):
    """Return the vector for `word` if in-vocabulary and non-zero, else None."""
    tok = nlp.vocab[word]
    if not tok.has_vector or tok.vector_norm == 0:
        return None
    return tok.vector


def run_spacy(args, rows):
    import spacy
    print(f"Loading spaCy model {args.spacy_model} ...", file=sys.stderr)
    nlp = spacy.load(args.spacy_model, disable=["parser", "ner", "tagger", "lemmatizer"])
    print("Loaded.", file=sys.stderr)
    return rows, lambda w: get_vector_spacy(nlp, w)


# ---------------------------------------------------------------------------
# gensim backend
# ---------------------------------------------------------------------------

def run_gensim(args, rows):
    from gensim.models import KeyedVectors
    print(f"Loading vectors from {args.vectors} ...", file=sys.stderr)
    binary = args.vectors.endswith(".bin")
    kv = KeyedVectors.load_word2vec_format(args.vectors, binary=binary)
    print("Loaded.", file=sys.stderr)

    def get_vector(word):
        if word in kv:
            return kv[word]
        return None

    return rows, get_vector


# ---------------------------------------------------------------------------

def process(rows, get_vector, fallback_to_prefix_only=True):
    """
    For each row, compute similarity between the correct reconstructed verb
    and each distractor reconstructed verb. Falls back to comparing just the
    bare prefix words (e.g. "an" vs "aus") if the full reconstructed verb is
    out-of-vocabulary — common for compositional separable verbs that a
    general-purpose vector model never saw as a single token.
    """
    out_rows = []
    n_full_word_hits = 0
    n_fallback_hits = 0
    n_total_pairs = 0

    for row in rows:
        stem = row["verb_stem"]
        correct_prefix = row["correct_prefix"]
        distractors = [d for d in row["distractor_prefixes"].split(";") if d]

        correct_verb = reconstruct_verb(correct_prefix, stem)
        correct_vec = get_vector(correct_verb)
        used_fallback_for_correct = False
        if correct_vec is None and fallback_to_prefix_only:
            correct_vec = get_vector(correct_prefix)
            used_fallback_for_correct = True

        sims = []
        methods = []
        for d in distractors:
            n_total_pairs += 1
            d_verb = reconstruct_verb(d, stem)
            d_vec = get_vector(d_verb)
            method = "full_verb"
            if d_vec is None and fallback_to_prefix_only:
                d_vec = get_vector(d)
                method = "prefix_only"
            if d_vec is None or correct_vec is None:
                sims.append(float("nan"))
                methods.append("no_vector")
                continue
            if method == "full_verb" and not used_fallback_for_correct:
                n_full_word_hits += 1
            else:
                n_fallback_hits += 1
            sims.append(cosine_sim(correct_vec, d_vec))
            methods.append(method)

        valid_sims = [s for s in sims if s == s]  # filter NaN
        max_sim = max(valid_sims) if valid_sims else float("nan")

        row_out = dict(row)
        row_out["distractor_similarities"] = ";".join(
            f"{s:.4f}" if s == s else "NA" for s in sims
        )
        row_out["similarity_methods"] = ";".join(methods)
        row_out["max_distractor_similarity"] = f"{max_sim:.4f}" if max_sim == max_sim else "NA"
        out_rows.append(row_out)

    print(f"\nVector lookup coverage: {n_full_word_hits} full-verb hits, "
          f"{n_fallback_hits} prefix-only fallback hits, "
          f"out of {n_total_pairs} distractor pairs.", file=sys.stderr)

    return out_rows


def assign_difficulty(rows):
    """
    Median-split items into 'hard' (close-meaning distractor, similarity
    above the median) vs 'easy' (below median), based on max_distractor_similarity.
    Items with no valid similarity score are labeled 'unknown'.
    """
    valid = [float(r["max_distractor_similarity"]) for r in rows
             if r["max_distractor_similarity"] != "NA"]
    if not valid:
        for r in rows:
            r["difficulty"] = "unknown"
        return rows

    valid.sort()
    median = valid[len(valid) // 2]

    for r in rows:
        if r["max_distractor_similarity"] == "NA":
            r["difficulty"] = "unknown"
        else:
            sim = float(r["max_distractor_similarity"])
            r["difficulty"] = "hard" if sim >= median else "easy"

    from collections import Counter
    print(f"Difficulty split (median similarity = {median:.4f}): "
          f"{dict(Counter(r['difficulty'] for r in rows))}", file=sys.stderr)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Stimulus CSV (from build_stimuli.py).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["spacy", "gensim"], default="spacy")
    parser.add_argument("--spacy-model", default="de_core_news_lg",
                         help="spaCy model to use (must have real vectors, i.e. the 'lg' size).")
    parser.add_argument("--vectors", default=None,
                         help="Path to a local word2vec/fastText vector file (gensim backend only).")
    parser.add_argument("--no-fallback", action="store_true",
                         help="Disable falling back to bare-prefix vectors when the full "
                              "reconstructed verb is out-of-vocabulary.")
    args = parser.parse_args()

    if args.backend == "gensim" and not args.vectors:
        parser.error("--vectors is required when --backend gensim is used.")

    with open(args.input, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} stimulus items from {args.input}", file=sys.stderr)

    if args.backend == "spacy":
        rows, get_vector = run_spacy(args, rows)
    else:
        rows, get_vector = run_gensim(args, rows)

    out_rows = process(rows, get_vector, fallback_to_prefix_only=not args.no_fallback)
    out_rows = assign_difficulty(out_rows)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()