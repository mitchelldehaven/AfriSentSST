"""
Script for reordering the multi outputs for scoring.
My approach uses lang_ids for setting NLLB tokenization,
so prediction is ran on all individual language files and
the results are concatenated into a single file. This process
requires re-ordering in order for the scoring to work properly.
"""
import pandas as pd
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv", type=Path, required=True)
    parser.add_argument("--source_tsv", type=Path, required=True)
    parser.add_argument("--output_tsv_path", type=Path, required=True)
    args = parser.parse_args()
    return args


def find_label(label_map, tweet):
    if tweet in label_map:
        return label_map[tweet]
    elif "#" + tweet in label_map:
        return label_map[tweet]
    else:
        return "neutral"

# def main():
if __name__ == "__main__":
    args = parse_args()
    input_df = pd.read_csv(args.input_tsv, sep="\t")
    source_df = pd.read_csv(args.source_tsv, sep="\t")
    tweet_label_map = dict(zip(input_df["tweet"].tolist(), input_df["label"].tolist()))
    count = 0
    total = 0
    for tweet in source_df["tweet"]:
        if tweet in tweet_label_map:
            count += 1
        # For some reason, some of the examples in the multilingual test set have an additional "#" on the front.
        elif "#" + tweet in tweet_label_map:
            count += 1
        total += 1
    print(f"Guessed on {(total - count) / total:.4%} ({total - count} / {total}) of examples")
    source_df["label"] = source_df["tweet"].apply(lambda x: tweet_label_map[x] if x in tweet_label_map else "neutral")
    del source_df["tweet"]
    source_df.to_csv(args.output_tsv_path, sep="\t", index=False)

# if __name__ == "__main__":
