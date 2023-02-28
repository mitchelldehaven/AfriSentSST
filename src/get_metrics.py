import argparse
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_tsv", type=Path, required=True)
    parser.add_argument("--pred_tsv", type=Path, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    gold_df = pd.read_csv(args.gold_tsv, sep="\t")
    pred_df = pd.read_csv(args.pred_tsv, sep="\t")
    f1 = f1_score(y_true = gold_df["label"], y_pred = pred_df["label"], average="weighted")
    recall = recall_score(y_true = gold_df["label"], y_pred = pred_df["label"], average="weighted")
    precision = precision_score(y_true = gold_df["label"], y_pred = pred_df["label"], average="weighted")
    print("Weight")
    print("-"*100)
    print("F1:", f1)
    print("Recall:", recall)
    print("Precision:", precision)
    print("="*100)
    f1 = f1_score(y_true = gold_df["label"], y_pred = pred_df["label"], average="micro")
    recall = recall_score(y_true = gold_df["label"], y_pred = pred_df["label"], average="micro")
    precision = precision_score(y_true = gold_df["label"], y_pred = pred_df["label"], average="micro")
    print("Micro")
    print("-"*100)
    print("F1:", f1)
    print("Recall:", recall)
    print("Precision:", precision)