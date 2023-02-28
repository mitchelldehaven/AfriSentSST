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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model_type", type=str, default="xlm-roberta-large")
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max_samples", type=int)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    model = Transformer(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    model.cuda()
    state_dict = torch.load(args.checkpoint, map_location=torch.device("cpu"))["state_dict"]
    if "model.loss_fct.weight" in state_dict:
        del state_dict["model.loss_fct.weight"]
    model.load_state_dict(state_dict)
    unsupervised_data = load_dataset(args.data_file, en_tweets=True, just_text=True)
    if args.max_samples:
        unsupervised_data = unsupervised_data[:args.max_samples]
    unsupervised_dataset = AfriSentDataset(unsupervised_data)
    partial_collate_fn = partial(collate_fn, tokenizer=tokenizer, max_length=128)
    unsupervised_dataloader = DataLoader(unsupervised_dataset, num_workers=2, batch_size=16, collate_fn=partial_collate_fn, shuffle=False)
    softmax_preds = []
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for batch_x, batch_y in tqdm(unsupervised_dataloader):
                to_cuda_hf(batch_x)
                outputs = model(batch_x)
                softmax_preds.append(F.softmax(outputs.logits.detach().cpu()))

    del unsupervised_dataloader
    # this is a bit memory inefficient
    softmax_preds = torch.cat(softmax_preds).float().numpy()
    df = pd.DataFrame(data=softmax_preds, columns=["negative_p", "neutral_p", "positive_p"])
    df["tweets"] = [d["text"] for d in unsupervised_data]

    df.to_csv(args.output_dir / "english_tweets_pseudo_label_scores.tsv", sep="\t", index=False)

