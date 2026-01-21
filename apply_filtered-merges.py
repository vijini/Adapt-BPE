import json
import argparse
from tokenizers import Tokenizer
from collections import Counter
from tqdm import tqdm


def load_filtered_merges(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_and_replace_merges(tokenizer_json_path, filtered_merges):
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tok = json.load(f)

    assert tok["model"]["type"] == "BPE"
    tok["model"]["merges"] = filtered_merges

    return tok


def tokenize_corpus(tokenizer, text_path):
    final_token_count = 0
    char_token_count = 0
    used_tokens = Counter()

    with open(text_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            line = line.rstrip("\n")
            if not line:
                continue

            # Character-level baseline
            char_token_count += len(line)

            enc = tokenizer.encode(line)

            # Final token count (after HF pre-tokenization + BPE merges)
            final_token_count += len(enc.tokens)

            for tok in enc.tokens:
                used_tokens[tok] += 1

    return char_token_count, final_token_count, used_tokens


def main(args):
    filtered_merges = load_filtered_merges(args.filtered_merges)

    tok_json = load_and_replace_merges(
        args.tokenizer_json, filtered_merges
    )

    tokenizer = Tokenizer.from_str(json.dumps(tok_json))

    char_token_count, final_token_count, used_tokens = tokenize_corpus(
        tokenizer, args.test_file
    )

    # Compression Utility (CU)
    cu = (char_token_count - final_token_count) / char_token_count

    merged_tokens = {
        merge.replace(" ", "")
        for merge in filtered_merges
    }

    used_merged_tokens = {
        t for t in used_tokens if t in merged_tokens
    }

    unused_merged_tokens = merged_tokens - used_merged_tokens

    print("===== Tokenization Statistics =====")
    print(f"Character-level token count: {char_token_count}")
    print(f"Final token count: {final_token_count}")
    print(f"Compression Utility (CU): {cu:.6f}")
    #print(f"Filtered merges: {len(filtered_merges)}")
    print(f"Used merged tokens: {len(used_merged_tokens)}")
    print(f"Unused merged tokens: {len(unused_merged_tokens)}")

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f"character_token_count\t{char_token_count}\n")
            f.write(f"final_token_count\t{final_token_count}\n")
            f.write(f"compression_utility\t{cu}\n")
           #f.write(f"filtered_merges\t{len(filtered_merges)}\n")
            f.write(f"used_merged_tokens\t{len(used_merged_tokens)}\n")
            f.write(f"unused_merged_tokens\t{len(unused_merged_tokens)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_json", required=True)
    parser.add_argument("--filtered_merges", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", default=None)

    args = parser.parse_args()
    main(args)
