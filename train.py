#!/usr/bin/env python3
import argparse
import logging
import json
from pathlib import Path
from collections import Counter, defaultdict
import copy

SEP = "\u0001"  

def tokenize_characters(text):
    """Character-level tokenization with </w> marking word boundaries.
       Returns list of token lists (per sentence word)."""
    words = text.strip().split()
    tokenized_words = []
    for word in words:
        if word:
            chars = list(word)
            chars.append("</w>")
            tokenized_words.append(chars)
    return tokenized_words

def serialize_word(token_list):
    """Serialize token list to a single string key."""
    return SEP.join(token_list)

def deserialize_word(word_key):
    """Back to token list."""
    if word_key == "":
        return []
    return word_key.split(SEP)

def load_merges_from_tokenizer_json(tokenizer_path):
    tokenizer_file = Path(tokenizer_path) / "tokenizer.json"
    with open(tokenizer_file, "r", encoding="utf-8") as f:
        tokenizer = json.load(f)
    merges = tokenizer["model"]["merges"]
    return [tuple(merge) for merge in merges]

def filter_merges(merges, num_valid_merges):
    """Same logic as original: build valid merges that are possible from characters."""
    valid_merges = []
    skipped_merges = []

    current_tokens = set()
    for a, b in merges:
        for token in (a, b):
            if len(token) == 1 or token == "</w>":
                current_tokens.add(token)

    for a, b in merges:
        if a in current_tokens and b in current_tokens:
            merged = a + b
            valid_merges.append((a, b))
            current_tokens.add(merged)
            if len(valid_merges) == num_valid_merges:
                break
        else:
            reason = []
            if a not in current_tokens:
                reason.append(f"'{a}' not in current_tokens")
            if b not in current_tokens:
                reason.append(f"'{b}' not in current_tokens")
            skipped_merges.append((a, b, reason))

    return valid_merges, skipped_merges

def get_initial_word_counts(tokenized_words):
    """Return Counter mapping serialized word -> count (weighted unique types)."""
    word_strings = [serialize_word(w) for w in tokenized_words]
    return Counter(word_strings)

def get_bigram_frequencies_from_counts(word_counts):
    """Compute bigram frequencies weighted by counts (pairs of token strings)."""
    freqs = defaultdict(int)
    for word_key, count in word_counts.items():
        tokens = deserialize_word(word_key)
        for i in range(len(tokens) - 1):
            freqs[(tokens[i], tokens[i + 1])] += count
    return freqs

def apply_merges_word_counts(word_counts, merges):
    """Apply merges on the word_counts Counter (efficient). Also return merge_frequencies list and corpus_size progression."""
    merge_frequencies = []
    compression_log = []

    # initial corpus size = total tokens excluding </w>, weighted by counts
    corpus_size = sum((len(deserialize_word(w)) - 1) * c for w, c in word_counts.items())

    for a, b in merges:
        merge_symbol = a + b
        replacements_this_round = 0
        new_word_counts = Counter()

        # For each unique word type, compute how many replacements will happen and the new word after merges
        for word_key, count in word_counts.items():
            tokens = deserialize_word(word_key)
            i = 0
            # We'll construct a new token sequence for this word
            out_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    out_tokens.append(merge_symbol)
                    replacements_this_round += count
                    i += 2
                else:
                    out_tokens.append(tokens[i])
                    i += 1
            new_key = serialize_word(out_tokens)
            new_word_counts[new_key] += count

        # update corpus size: each replacement reduces token count by 1 per occurrence (excluding </w> hasn't changed)
        corpus_size_after = corpus_size - replacements_this_round
        corpus_size = corpus_size_after

        merge_frequencies.append(((a, b), replacements_this_round))
        compression_log.append((a, b, replacements_this_round, corpus_size_after))

        word_counts = new_word_counts

    return word_counts, merge_frequencies, compression_log

def undo_merge_word_counts(word_counts, merge):
    """Undo a merge in word_counts: replace merged token with (a,b) split.
       Returns a new Counter (no in-place mutate for safety)."""
    a, b = merge
    merged = a + b
    new_counts = Counter()
    for word_key, count in word_counts.items():
        tokens = deserialize_word(word_key)
        # if merged not in tokens, keep as is
        if merged not in tokens:
            new_counts[word_key] += count
            continue

        # otherwise, split every occurrence of merged into [a, b]
        # build new token list
        out_tokens = []
        for tok in tokens:
            if tok == merged:
                out_tokens.append(a)
                out_tokens.append(b)
            else:
                out_tokens.append(tok)
        new_key = serialize_word(out_tokens)
        new_counts[new_key] += count

    return new_counts

def refine_merges_word_counts(tokenized_word_counts, applied_merges, remaining_merges, merge_frequencies):
    """Preserve original refinement algorithm but operate on word_counts for efficiency."""
    log_data = []

    # applied_sorted: sort applied merges by their recorded frequency (ascending)
    applied_sorted = sorted(merge_frequencies, key=lambda x: x[1])  # [(merge, freq), ...]
    # current bigram freqs from the current tokenized representation
    current_bigram_freqs = get_bigram_frequencies_from_counts(tokenized_word_counts)

    remaining_merge_set = set(remaining_merges)
    # filter bigrams to only those present in remaining merges
    filtered_bigrams = [
        (bigram, freq) for bigram, freq in current_bigram_freqs.items()
        if bigram in remaining_merge_set
    ]
    remaining_sorted = sorted(filtered_bigrams, key=lambda x: x[1], reverse=True)  # high freq first

    merge_number = 1
    final_applied_merges = applied_merges.copy()
    word_counts = tokenized_word_counts

    # For stable behavior similar to original: work until we exhaust applied_sorted or remaining_sorted
    while applied_sorted:
        low_merge, low_freq = applied_sorted.pop(0)  # lowest-frequency applied merge
        low_a, low_b = low_merge

        # skip any remaining_sorted that have freq <= low_freq (original code removed them)
        while remaining_sorted and remaining_sorted[0][1] <= low_freq:
            _skipped = remaining_sorted.pop(0)

        if not remaining_sorted:
            break

        high_merge, high_freq = remaining_sorted.pop(0)
        high_a, high_b = high_merge

        # Candidate = undo low_merge, then try to apply high_merge and count replacements
        candidate_counts = undo_merge_word_counts(word_counts, low_merge)

        # apply high_merge on candidate_counts (count replacements)
        replacements = 0
        new_candidate_counts = Counter()
        for word_key, count in candidate_counts.items():
            tokens = deserialize_word(word_key)
            i = 0
            out_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == high_a and tokens[i + 1] == high_b:
                    out_tokens.append(high_a + high_b)
                    replacements += count
                    i += 2
                else:
                    out_tokens.append(tokens[i])
                    i += 1
            new_key = serialize_word(out_tokens)
            new_candidate_counts[new_key] += count

        if replacements > 0:
            # Accept replacement: update word_counts and final_applied_merges
            word_counts = new_candidate_counts
            # remove low_merge from final_applied_merges and append high_merge
            # keep order: remove first occurrence of low_merge
            try:
                final_applied_merges.remove(low_merge)
            except ValueError:
                # if not found, skip
                pass
            final_applied_merges.append(high_merge)

            log_data.append({
                "merge_type": "Replacement",
                "number": merge_number,
                "low_merge": low_merge,
                "low_freq": low_freq,
                "high_merge": high_merge,
                "high_freq": high_freq,
                "replacements": replacements
            })
            merge_number += 1
        # else: no effect; continue

    return word_counts, log_data, final_applied_merges

def flatten_from_counts(word_counts):
    """Flatten word_counts into final string without </w> tokens."""
    final_tokens = []
    for word_key, count in word_counts.items():
        tokens = deserialize_word(word_key)
        # remove </w> tokens and extend count times
        tokens_no_w = [t for t in tokens if t != "</w>"]
        for _ in range(count):
            final_tokens.extend(tokens_no_w)
    return " ".join(final_tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--num_merges", type=int, required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--final_merge_file", required=True)
    args = parser.parse_args()

    # Setup logging to file (but we will batch messages)
    logging.basicConfig(filename=args.log_file, level=logging.INFO, format="%(message)s")
    logs = []
    logs.append("=== Adapt-BPE Started ===")

    text = Path(args.test_file).read_text(encoding="utf-8")
    tokenized_words = tokenize_characters(text)
    total_chars = sum(len(w) for w in tokenized_words)
    logs.append(f"Corpus size (chars): {total_chars}")

    # Build word counts (efficient)
    word_counts = get_initial_word_counts(tokenized_words)
    logs.append(f"Unique word types: {len(word_counts)}  Total tokens (approx): {sum((len(deserialize_word(w)) - 1) * c for w, c in word_counts.items())}")

    all_merges = load_merges_from_tokenizer_json(args.tokenizer_path)
    applied_merges, skipped_merges = filter_merges(all_merges, args.num_merges)
    logs.append(f"Filtered {len(applied_merges)} merges to apply.")

    if skipped_merges:
        for a, b, reasons in skipped_merges:
            logs.append(f"Skipped Merge: ({a},{b}) Reason: {', '.join(reasons)}")

    # Apply merges (efficiently) while collecting merge frequencies and compression log
    word_counts, merge_frequencies, compression_log = apply_merges_word_counts(word_counts, applied_merges)

    # remaining merges list for refinement
    remaining_merges = [m for m in all_merges if m not in applied_merges]

    # Now run refinement (same logic as original) but efficient
    word_counts, refinement_log, final_applied_merges = refine_merges_word_counts(
        word_counts, applied_merges, remaining_merges, merge_frequencies
    )

    final_text = flatten_from_counts(word_counts)

    # Write outputs
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.final_merge_file).parent.mkdir(parents=True, exist_ok=True)

    Path(args.output_file).write_text(final_text, encoding="utf-8")
    with open(args.final_merge_file, "w", encoding="utf-8") as f:
        for a, b in final_applied_merges:
            f.write(f"{a} {b}\n")

    # write logs (batched)
    for line in logs:
        logging.info(line)
    for entry in compression_log:
        logging.info(f"COMP_LOG: {entry}")
    for r in refinement_log:
        logging.info(f"REFINE: {r}")

    logging.info("=== Adapt-BPE Completed ===")

if __name__ == "__main__":
    main()
