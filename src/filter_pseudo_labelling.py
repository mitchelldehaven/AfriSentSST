from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.models.transformer import ECELoss, Transformer
from torch.utils.data import DataLoader
import sys
import torch
from src.util import to_cuda_hf
from src.paths import DATA_DIR
import pandas as pd

import os
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import random
import argparse
from pathlib import Path
from functools import partial
from transformers import AutoTokenizer
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
import math
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from src.paths import ROOT_DIR, MODELS_DIR
from src.util import load_dataset
from src.models import Transformer
from src.models.dataset import AfriSentDataset, collate_fn
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from tqdm import tqdm

import pandas as pd
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--negative_threshold", type=float, default=0.9)
    parser.add_argument("--neutral_threshold", type=float, default=0.9)
    parser.add_argument("--positive_threshold", type=float, default=0.9)
    args = parser.parse_args()
    return args


def is_clean_sample(text):
    og_splits = text.split()
    text = re.sub(r"@[^\s]*", " ", text)
    text = re.sub(r"http[^\s]*", " ", text)
    splits = text.split()
    return len(splits) > 2 and len(splits) / len(og_splits) >= 0.6


if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.data_file, delimiter="\t").dropna()
    negative_samples = df[df["negative_p"] > args.negative_threshold]["tweets"]
    neutral_samples = df[df["neutral_p"] > args.neutral_threshold]["tweets"]
    positive_samples = df[df["positive_p"] > args.positive_threshold]["tweets"]
    print(len(negative_samples), len(neutral_samples), len(positive_samples))
    negative_samples = [negative_sample for negative_sample in negative_samples if is_clean_sample(negative_sample)]
    neutral_samples = [neutral_sample for neutral_sample in neutral_samples if is_clean_sample(neutral_sample)]
    positive_samples = [positive_sample for positive_sample in positive_samples if is_clean_sample(positive_sample)]
    with open(args.output_dir / f"english_tweets_pseudo_labeled_thresholds_{args.negative_threshold}_{args.neutral_threshold}_{args.positive_threshold}.tsv", "w") as f:
        i = 0
        print("ID\tlabel\ttweet", file=f)
        for negative_sample in negative_samples:
            print("\t".join([str(s) for s in [i, "negative", negative_sample]]), file=f)
            i += 1
        for neutral_sample in neutral_samples:
            print("\t".join([str(s) for s in [i, "neutral", neutral_sample]]), file=f)
            i += 1
        for positive_sample in positive_samples:
            print("\t".join([str(s) for s in [i, "positive", positive_sample]]), file=f)
            i += 1

    print(len(negative_samples), len(neutral_samples), len(positive_samples))
