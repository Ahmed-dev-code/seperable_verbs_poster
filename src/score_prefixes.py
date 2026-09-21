"""
score_prefixes.py

Computes, for each item in stimulus_set_v1.csv, how strongly a language model
prefers the CORRECT separable-verb prefix over DISTRACTOR prefixes, given the
truncated German sentence context.

Two backends are supported:
  1. "hf"  - a local Hugging Face causal LM (e.g. a German/multilingual GPT-style model)
  2. "api" - the Anthropic API (or any OpenAI-compatible endpoint) that returns logprobs

For each item and each candidate prefix (correct + distractors), we compute the
model's log-probability of that prefix's tokens, teacher-forced, right after the
context. The prefix is prepended with a leading space (since German writes a space
before it) to match how it appears in real text.

Usage (HF backend):
    python score_prefixes.py --backend hf --model dbmdz/german-gpt2 \
        --input stimulus_set_v1.csv --output results_german-gpt2.csv

Usage (API backend, if your provider supports logprobs / echo):
    python score_prefixes.py --backend api --model <model-name> \
        --input stimulus_set_v1.csv --output results_api.csv
"""

import argparse
import csv
import math
import sys


# ---------------------------------------------------------------------------
# HF backend
# ---------------------------------------------------------------------------

def score_candidate_hf(model, tokenizer, context, candidate, device):
    """
    Return the total log-probability (natural log, teacher-forced) that the model
    assigns to `candidate` (a single word, e.g. a separable prefix) appearing
    immediately after `context`.

    We prepend a space to the candidate since that's how it would appear in the
    real sentence, then tokenize context and context+candidate separately to
    identify exactly which tokens belong to the candidate.
    """
    import torch

    context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    full_text = context + " " + candidate
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    n_context_tokens = context_ids.shape[1]
    n_full_tokens = full_ids.shape[1]

    if n_full_tokens <= n_context_tokens:
        # Degenerate case: candidate contributed no new tokens (rare, but guard for it)
        return float("nan"), 0

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # shape: (1, seq_len, vocab_size)

    log_probs = torch.log_softmax(logits, dim=-1)

    total_logprob = 0.0
    n_candidate_tokens = n_full_tokens - n_context_tokens
    for pos in range(n_context_tokens, n_full_tokens):
        # logits at position (pos - 1) predict the token at position `pos`
        target_token_id = full_ids[0, pos]
        token_logprob = log_probs[0, pos - 1, target_token_id].item()
        total_logprob += token_logprob

    return total_logprob, n_candidate_tokens


def run_hf(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {args.model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}.", file=sys.stderr)

    rows = list(csv.DictReader(open(args.input, encoding="utf-8")))
    fieldnames = list(rows[0].keys()) + [
        "correct_logprob",
        "correct_logprob_per_token",
        "distractor_logprobs",       # semicolon-separated, aligned with distractor_prefixes
        "best_distractor_logprob",
        "margin",                    # correct_logprob - best_distractor_logprob
        "model_prefers_correct",     # 1 if margin > 0 else 0 (i.e. rank == 1)
        "rank",                      # 1 = correct prefix scored highest among all candidates
        "n_candidates",              # total number of candidates it competed against (incl. correct)
        "reciprocal_rank",           # 1/rank -- standard ranking metric (MRR when averaged)
    ]

    out_rows = []
    for i, row in enumerate(rows):
        context = row["context_sentence"]
        correct = row["correct_prefix"]
        distractors = [d for d in row["distractor_prefixes"].split(";") if d]

        correct_lp, correct_ntok = score_candidate_hf(model, tokenizer, context, correct, device)
        correct_lp_per_tok = correct_lp / correct_ntok if correct_ntok else float("nan")

        distractor_lps = []
        for d in distractors:
            d_lp, _ = score_candidate_hf(model, tokenizer, context, d, device)
            distractor_lps.append(d_lp)

        best_distractor_lp = max(distractor_lps) if distractor_lps else float("-inf")
        margin = correct_lp - best_distractor_lp

        # Rank: sort all candidates (correct + distractors) by logprob descending,
        # find the 1-indexed position of the correct one. Ties broken in favor of
        # the correct candidate (i.e. a tie counts as rank 1) to avoid penalizing
        # exact float equality edge cases.
        all_scores = [("__correct__", correct_lp)] + list(zip(distractors, distractor_lps))
        all_scores.sort(key=lambda x: x[1], reverse=True)
        rank = next(i + 1 for i, (label, _) in enumerate(all_scores) if label == "__correct__")
        n_candidates = len(all_scores)
        reciprocal_rank = 1.0 / rank

        row_out = dict(row)
        row_out["correct_logprob"] = correct_lp
        row_out["correct_logprob_per_token"] = correct_lp_per_tok
        row_out["distractor_logprobs"] = ";".join(f"{x:.4f}" for x in distractor_lps)
        row_out["best_distractor_logprob"] = best_distractor_lp
        row_out["margin"] = margin
        row_out["model_prefers_correct"] = int(rank == 1)
        row_out["rank"] = rank
        row_out["n_candidates"] = n_candidates
        row_out["reciprocal_rank"] = reciprocal_rank
        out_rows.append(row_out)

        if (i + 1) % 20 == 0:
            print(f"  scored {i + 1}/{len(rows)} items", file=sys.stderr)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    n_correct = sum(r["model_prefers_correct"] for r in out_rows)
    mean_rr = sum(r["reciprocal_rank"] for r in out_rows) / len(out_rows)
    mean_rank = sum(r["rank"] for r in out_rows) / len(out_rows)
    print(f"\nDone. Top-1 accuracy: {n_correct}/{len(out_rows)} "
          f"({100 * n_correct / len(out_rows):.1f}%)")
    print(f"Mean reciprocal rank (MRR): {mean_rr:.3f}")
    print(f"Mean rank: {mean_rank:.2f} (out of avg {sum(r['n_candidates'] for r in out_rows) / len(out_rows):.1f} candidates)")
    print(f"Saved detailed results to {args.output}")


# ---------------------------------------------------------------------------
# API backend (Anthropic Messages API does not expose token logprobs;
# this stub targets an OpenAI-compatible /completions-style endpoint with
# `echo=True, logprobs=N`, which several providers/self-hosted servers support.
# Adjust `call_api_for_logprobs` to match whatever provider you actually use.
# ---------------------------------------------------------------------------

def call_api_for_logprobs(client, model, full_text, n_context_chars):
    """
    Placeholder: call your API with echo+logprobs enabled and return a list of
    (token_str, token_logprob, token_start_char) for the full_text.
    You will need to adapt this to your provider's actual response format.
    """
    raise NotImplementedError(
        "Fill this in for your specific API provider. Most providers that expose "
        "token-level logprobs support an 'echo' or 'completions' style call. "
        "The Anthropic Messages API does not currently expose input-token logprobs, "
        "so for API-based scoring you'll likely need an OpenAI-compatible endpoint "
        "or a self-hosted inference server (e.g. vLLM, TGI) that does."
    )


def run_api(args):
    print(
        "API backend is a template — see call_api_for_logprobs() in this script "
        "and fill in the request/response handling for your provider.",
        file=sys.stderr,
    )
    raise NotImplementedError


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "api"], required=True)
    parser.add_argument("--model", required=True,
                         help="HF model id (e.g. dbmdz/german-gpt2, benjamin/gerpt2, "
                              "or a multilingual model) or API model name.")
    parser.add_argument("--input", default="stimulus_set_v1.csv")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    if args.backend == "hf":
        run_hf(args)
    else:
        run_api(args)


if __name__ == "__main__":
    main()