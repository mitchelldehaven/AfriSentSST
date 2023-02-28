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
import src.util as utils


def init_lang_tokens(tokenizer):
    for lang_id in utils.lang_ids:
            tokenizer.add_tokens(f"<{lang_id}>")

def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint_steps", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulations", type=int, default=1)
    parser.add_argument("--tokenizer_max_length", type=int, default=128)
    parser.add_argument("--model_type", type=str, default="xlm-roberta-large")
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--random_seeds", type=int, default=[0], nargs="+")
    parser.add_argument("--sst_files", type=Path, nargs="+")
    parser.add_argument("--use_lang_id_token", action="store_true")
    parser.add_argument("--save_dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--validation_percent", type=float, default=0.1)
    parser.add_argument("--dev_as_validation", action="store_true")
    args = parser.parse_args()
    return args


def train(args):
    args.save_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_accuracy',
        dirpath=MODELS_DIR,
        filename='multi_{epoch:02d}-{valid_accuracy:.5f}',
        save_top_k=1,
        mode='max',
        verbose=True,
    )   
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    print(len(tokenizer))
    for seed in args.random_seeds:
        pl.seed_everything(seed)
        lr_callback = LearningRateMonitor(logging_interval="step")
        train_datasets = []
        valid_datasets = []
        for lang_id in utils.lang_ids:
            if args.use_lang_id_token:
                tokenizer.add_tokens(f"<{lang_id}>")
            random_seed = random.Random(seed)
            train_data = load_dataset(ROOT_DIR / "SubtaskA" / "train" / f"{lang_id}_train.tsv")
            random_seed.shuffle(train_data)
            if args.dev_as_validation:
                valid_data = load_dataset(ROOT_DIR / "SubtaskA" / "dev_gold" / f"{lang_id}_dev_gold_label.tsv")
            else:
                valid_size = int(len(train_data) * args.validation_percent)
                valid_data = train_data[-valid_size:]
                train_data = train_data[:-valid_size]
            lang_id = lang_id if args.use_lang_id_token else None
            train_datasets.append(AfriSentDataset(train_data, lang_id=lang_id))
            valid_datasets.append(AfriSentDataset(valid_data, lang_id=lang_id))
        partial_collate_fn = partial(collate_fn, tokenizer=tokenizer, max_length=args.tokenizer_max_length)
        train_dataloader = DataLoader(
            torch.utils.data.ConcatDataset(train_datasets), 
            batch_size=args.batch_size, 
            num_workers=1, 
            collate_fn=partial_collate_fn, 
            shuffle=True)
        valid_dataloader = DataLoader(
            torch.utils.data.ConcatDataset(valid_datasets), 
            batch_size=2*args.batch_size,
            num_workers=1,
            collate_fn=partial_collate_fn)
        print(len(tokenizer))
        steps_per_epoch = math.ceil((len(torch.utils.data.ConcatDataset(train_datasets)) / args.batch_size) / args.gradient_accumulations)
        model = Transformer(args.model_type, tokenizer, steps_per_epoch=steps_per_epoch, epochs=args.epochs, lr=args.lr)
        trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, default_root_dir="checkpoints", precision="bf16",
                            callbacks=[lr_callback, checkpoint_callback], accumulate_grad_batches=args.gradient_accumulations, 
                            val_check_interval=1.0)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
 

if __name__ == "__main__":
    args = parse_training_args()
    train(args)