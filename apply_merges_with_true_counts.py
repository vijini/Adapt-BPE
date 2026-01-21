#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import Counter

SEP = "\u0001"   # separator for serialization
EOW = "</w>"     # end-of-word marker


# Token helpers
def make_base_token(s):
    return s

def make_merged_token(a, b):
    return a + b



# Tokenization (character-level)
def tokenize_text(text):
    """
    Character-level tokenization with </w>.
    Returns list of tokenized words.
    """
    words = text.strip().split()
    tokenized = []
    for w in words:
        chars = [make_base_token(c) for c in w]
        chars.append(EOW)
        tokenized.append(chars)
    return tokenized


# Serialization helpers
def serialize(tokens):
    return SEP.join(tokens)

def deserialize(s):
    return s.split(SEP)


# Apply merges with counting
def apply_merges(word_counts, merges):
    """
    Applies merges sequentially.
    Tracks how many times each merge is applied.
    """
    merge_usage = Counter()

    for merge_id, (a, b) in enumerate(merges):
        new_counts = Counter()

        for word_key, count in word_counts.items():
            tokens = deserialize(word_key)
            i = 0
            out = []

            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    out.append(make_merged_token(a, b))
                    merge_usage[(a, b)] += count
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1

            new_counts[serialize(out)] += count

        word_counts = new_counts

    return word_counts, merge_usage


# Count final tokens
def count_final_tokens(word_counts):
    total = 0
    for word_key, count in word_counts.items():
        tokens = deserialize(word_key)
        tokens = [t for t in tokens if t != EOW]
        total += len(tokens) * count
    return total


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--merge_file", required=True,
                        help="One merge per line: 'a b'")
    parser.add_argument("--output_merge_usage", required=True)
    parser.add_argument("--output_stats", required=True)
    args = parser.parse_args()

   
    merges = []
    with open(args.merge_file, "r", encoding="utf-8") as f:
        for line in f:
            a, b = line.strip().split()
            merges.append((a, b))

  
    text = Path(args.test_file).read_text(encoding="utf-8")

   
    char_count = sum(len(line.rstrip("\n")) for line in text.splitlines())

    
    tokenized_words = tokenize_text(text)
    word_counts = Counter(serialize(w) for w in tokenized_words)

    
    final_counts, merge_usage = apply_merges(word_counts, merges)

    
    final_token_count = count_final_tokens(final_counts)

    
    cu = (char_count - final_token_count) / char_count

  
    with open(args.output_merge_usage, "w", encoding="utf-8") as f:
        f.write("merge\ttimes_used\n")
        for a, b in merges:
            f.write(f"{a} {b}\t{merge_usage.get((a,b), 0)}\n")

    
    with open(args.output_stats, "w", encoding="utf-8") as f:
        f.write(f"character_token_count\t{char_count}\n")
        f.write(f"final_token_count\t{final_token_count}\n")
        f.write(f"compression_utility\t{cu:.6f}\n")

    print("=== Done ===")
    print(f"CU: {cu:.6f}")
    print(f"Final tokens: {final_token_count}")


if __name__ == "__main__":
    main()
