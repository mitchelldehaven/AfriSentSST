import sqlite3
from pathlib import Path
import spacy_fastlang
from src.paths import DATA_DIR
import spacy
import argparse


NLP = spacy.load("en_core_web_sm")
NLP.add_pipe("language_detector")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_file", type=Path)
    parser.add_argument("--db_candidates", type=int, default=20_000_000)
    parser.add_argument("--english_candidates", type=int, default=5_000_000)
    parser.add_argument("--num_spacy_processes", type=int, default=8)
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    conn = sqlite3.connect(args.db_file)
    cmd = f"SELECT tweet FROM tweets LIMIT {args.db_candidates}"
    res = conn.execute(cmd)
    results = [t[0] for t in res.fetchmany(10_000)]
    english_tweets = []
    non_english_tweets = []
    while results:
        for doc_tweet in NLP.pipe(results, batch_size=1000, n_process=args.num_spacy_processes):
            if doc_tweet._.language == 'en':
                english_tweets.append(str(doc_tweet))
            else:
                non_english_tweets.append(str(doc_tweet))
            if len(english_tweets) >= args.english_candidates:
                break
        if len(english_tweets) >= args.english_candidates:
            break
        print(len(english_tweets))
        print(len(non_english_tweets))
        results = [t[0] for t in res.fetchmany(10_000)]

    conn.close()
            
    with open(args.output_dir / "english_tweets.txt", "w") as f:
        for english_tweet in english_tweets:
            print(english_tweet.replace("\n", " "), file=f)
    
    with open(args.output_dir / "non_english_tweets.txt", "w") as f:
        for non_english_tweet in non_english_tweets:
            print(non_english_tweet.replace("\n", " "), file=f)