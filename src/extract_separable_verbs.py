"""
extract_separable_verbs.py

Extracts every separable-verb-prefix construction (UD relation `compound:prt`)
from one or more CoNLL-U files, with all fields needed for the full study:
stimulus building, distance analysis, and manual review.

Usage:
    python extract_separable_verbs.py data/raw/UD_German-HDT/de_hdt-ud-dev.conllu
    python extract_separable_verbs.py data/raw/UD_German-HDT/*.conllu data/raw/UD_German-GSD/*.conllu -o data/extracted/separable_verbs_extracted.csv

On Windows/VS Code, glob patterns aren't expanded by the shell automatically in
all terminals (PowerShell does, cmd.exe doesn't) — you can also just list files
explicitly, or use the --glob flag to expand patterns yourself, e.g.:
    python extract_separable_verbs.py --glob "data\\raw\\**\\*.conllu"

Output columns:
    source_file, sent_id, text
    verb_id, verb_form, verb_lemma, verb_xpos, verb_feats
    prefix_id, prefix_form, prefix_lemma, prefix_xpos
    distance                -> number of tokens strictly between verb and prefix
    sentence_len             -> total token count in the sentence
    intervening_text          -> the actual words between verb and prefix (for manual review)
    n_interfering_prefixes    -> count of OTHER verb-particle (PTKVZ) tokens sitting in the
                                 intervening material — a competing separable-verb prefix
                                 (belonging to some other verb) that appears before the model
                                 reaches the target prefix. A potential confound independent
                                 of raw distance.
    interfering_prefix_forms  -> the actual interfering particle word(s), if any
    reconstructed_lemma       -> prefix_lemma + verb_lemma (e.g. "an" + "kommen" -> "ankommen")
    context_up_to_prefix      -> sentence text truncated to end right before the prefix
                                 (this is what you feed a model for scoring; the prefix
                                 itself and everything after it is excluded)
"""

import argparse
import csv
import glob
import sys
from pathlib import Path


def read_conllu_sentences(path):
    """
    Yield one dict per sentence:
        {
            "sent_id": str or None,
            "text": str or None,        # from the '# text = ...' comment, if present
            "tokens": [ {id, form, lemma, upos, xpos, feats, head, deprel}, ... ]
        }
    Multiword tokens (e.g. "1-2") and empty nodes (e.g. "8.1") are skipped for
    indexing purposes, matching standard UD conventions for dependency analysis.
    """
    sent_id = None
    text = None
    tokens = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("# sent_id"):
                sent_id = line.split("=", 1)[1].strip()
                continue
            if line.startswith("# text"):
                text = line.split("=", 1)[1].strip()
                continue
            if line.startswith("#"):
                continue

            if line.strip() == "":
                if tokens:
                    yield {"sent_id": sent_id, "text": text, "tokens": tokens}
                sent_id, text, tokens = None, None, []
                continue

            cols = line.split("\t")
            if len(cols) != 10:
                continue
            tok_id = cols[0]
            if "-" in tok_id or "." in tok_id:
                continue  # multiword token span or empty node

            tokens.append({
                "id": int(tok_id),
                "form": cols[1],
                "lemma": cols[2],
                "upos": cols[3],
                "xpos": cols[4],
                "feats": cols[5],
                "head": int(cols[6]) if cols[6].isdigit() else None,
                "deprel": cols[7],
            })

        if tokens:
            yield {"sent_id": sent_id, "text": text, "tokens": tokens}


def build_context_and_reconstruction(sentence_tokens, verb_tok, prefix_tok, fallback_text):
    """
    Build the truncated context string (everything up to but not including the
    prefix token). Prefers reconstructing from the raw token forms (respecting
    SpaceAfter=No where present would require MISC field parsing, which HDT/GSD
    mostly don't rely on for this purpose) — falls back to naive space-joining.
    We use simple space-joining here since German UD treebanks are consistently
    space-separated at the word level for this construction type.
    """
    ordered = sorted(sentence_tokens, key=lambda t: t["id"])
    context_tokens = [t["form"] for t in ordered if t["id"] < prefix_tok["id"]]
    return " ".join(context_tokens)


def extract_from_file(path):
    rows = []
    for sent in read_conllu_sentences(path):
        tokens = sent["tokens"]
        by_id = {t["id"]: t for t in tokens}

        for tok in tokens:
            if tok["deprel"] != "compound:prt":
                continue
            verb = by_id.get(tok["head"])
            prefix = tok
            if verb is None:
                continue
            # keep only the canonical order: verb (finite, earlier) ... prefix (later)
            if verb["id"] >= prefix["id"]:
                continue

            intervening = [t for t in sorted(tokens, key=lambda x: x["id"])
                            if verb["id"] < t["id"] < prefix["id"]]
            distance = len(intervening)
            intervening_text = " ".join(t["form"] for t in intervening)

            # Interference check: are there OTHER verb-particle tokens (PTKVZ)
            # sitting in the intervening material? These belong to some other
            # verb (e.g. an embedded clause or coordinated verb) but could
            # still act as a distractor/confound independent of raw distance.
            interfering = [t for t in intervening if t["xpos"] == "PTKVZ"]
            n_interfering_prefixes = len(interfering)
            interfering_prefix_forms = ";".join(t["form"] for t in interfering)

            context = build_context_and_reconstruction(tokens, verb, prefix, sent["text"])

            rows.append({
                "source_file": Path(path).name,
                "sent_id": sent["sent_id"],
                "text": sent["text"] or " ".join(t["form"] for t in sorted(tokens, key=lambda x: x["id"])),
                "verb_id": verb["id"],
                "verb_form": verb["form"],
                "verb_lemma": verb["lemma"],
                "verb_xpos": verb["xpos"],
                "verb_feats": verb["feats"],
                "prefix_id": prefix["id"],
                "prefix_form": prefix["form"],
                "prefix_lemma": prefix["lemma"],
                "prefix_xpos": prefix["xpos"],
                "distance": distance,
                "sentence_len": len(tokens),
                "intervening_text": intervening_text,
                "n_interfering_prefixes": n_interfering_prefixes,
                "interfering_prefix_forms": interfering_prefix_forms,
                "reconstructed_lemma": prefix["lemma"] + verb["lemma"],
                "context_up_to_prefix": context,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*",
                         help="One or more .conllu file paths (shell-expanded globs work on "
                              "most systems/terminals, e.g. bash or PowerShell).")
    parser.add_argument("--glob", action="append", default=[],
                         help="A glob pattern to expand manually (useful on Windows cmd.exe, "
                              "which does not expand wildcards itself). Can be given multiple times.")
    parser.add_argument("-o", "--output", default="data/extracted/separable_verbs_extracted.csv",
                         help="Output CSV path.")
    args = parser.parse_args()

    files = list(args.inputs)
    for pattern in args.glob:
        files.extend(glob.glob(pattern, recursive=True))

    # de-duplicate while preserving order
    seen = set()
    files = [f for f in files if not (f in seen or seen.add(f))]

    if not files:
        parser.error("No input files given. Pass one or more .conllu paths, or use --glob.")

    print(f"Processing {len(files)} file(s):", file=sys.stderr)
    for f in files:
        print(f"  {f}", file=sys.stderr)

    all_rows = []
    for f in files:
        rows = extract_from_file(f)
        print(f"  -> {len(rows)} separable-verb instances from {Path(f).name}", file=sys.stderr)
        all_rows.extend(rows)

    if not all_rows:
        print("No compound:prt instances found in the given file(s).", file=sys.stderr)
        sys.exit(1)

    fieldnames = list(all_rows[0].keys())
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nTotal instances extracted: {len(all_rows)}", file=sys.stderr)
    print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()