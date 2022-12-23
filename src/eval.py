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
from src.paths import ROOT_DIR, MODELS_DIR, OUTPUTS_DIR
from src.util import load_dataset, to_cuda_hf, dump_predictions
from src.models import Transformer
from src.models.dataset import AfriSentDataset, collate_fn

import pandas as pd
from tqdm import tqdm


def parse_inference_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_id", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tokenizer_max_length", type=int, default=128)
    parser.add_argument("--model_type", type=str, default="xlm-roberta-large")
    parser.add_argument("--random_seed", type=int, default=0)
    args = parser.parse_args()
    return args


def evaluate(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    model = Transformer(args.model_type)
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt["state_dict"])
    model.cuda()
    dev_file = ROOT_DIR / "SubtaskA" / "dev" / f"{args.lang_id}_dev.tsv"
    output_file = OUTPUTS_DIR / f"pred_{args.lang_id}.tsv"
    dev_pd = pd.read_csv(dev_file, sep="\t")
    dev_data = load_dataset(dev_file)
    dev_dataset = AfriSentDataset(dev_data)
    id2label = {v: k for k, v in dev_dataset.label2id.items()}
    partial_collate_fn = partial(collate_fn, tokenizer=tokenizer, max_length=args.tokenizer_max_length)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=partial_collate_fn)
    predictions = []
    with torch.no_grad():
        for batch_inputs, batch_labels in tqdm(dev_dataloader):
            to_cuda_hf(batch_inputs)
            batch_outputs = model(batch_inputs)
            batch_predictions = batch_outputs.logits.argmax(axis=1).detach().cpu()
            for pred in batch_predictions:
                pred_str = id2label[pred.item()]
                predictions.append(pred_str)
    
    dump_predictions(dev_file, output_file, predictions)


if __name__ == "__main__":
    args = parse_inference_args()
    evaluate(args) 