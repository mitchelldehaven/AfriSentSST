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




def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_id", type=str, required=True)
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
    parser.add_argument("--validation_file", type=Path)
    parser.add_argument("--train_file", type=Path)
    parser.add_argument("--save_dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--normalize_class_dist", action="store_true")
    parser.add_argument("--label_smoothing", default=0, type=float)
    parser.add_argument("--sst_file", type=Path)
    parser.add_argument("--validation_interval", type=float, default=1.0)
    parser.add_argument("--remove_handles_urls", action="store_true")
    parser.add_argument("--validation_percent", type=float, default=0.1)
    parser.add_argument("--dev_as_validation", action="store_true")
    args = parser.parse_args()
    return args


def train(args):
    args.save_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_accuracy',
        dirpath=args.save_dir,
        filename=f"{args.lang_id}_{args.model_type.replace('/', '--')}" + '_{epoch:02d}_{valid_accuracy:.5f}',
        save_top_k=1,
        mode='max',
        verbose=True,
    )        
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=torch.device("cpu"))["state_dict"]
        if "model.loss_fct.weight" in state_dict:
            del state_dict["model.loss_fct.weight"]
    for seed in args.random_seeds:
        lr_callback = LearningRateMonitor(logging_interval="step")
        pl.seed_everything(seed)
        random_seed = random.Random(seed)
        if args.train_file:
            train_data = load_dataset(args.train_file, en_tweets=True if args.lang_id == "en" else False)
        else:
            train_data = load_dataset(ROOT_DIR / "SubtaskA" / "train" / f"{args.lang_id}_train.tsv", en_tweets=True if args.lang_id == "en" else False)
        if args.validation_file:
            valid_data = load_dataset(args.validation_file, en_tweets=True if args.lang_id == "en" else False)
        elif args.dev_as_validation:
            valid_data = load_dataset(ROOT_DIR / "SubtaskA" / "dev_gold" / f"{args.lang_id}_dev_gold_label.tsv", en_tweets=True if args.lang_id == "en" else False)
        else:
            random_seed.shuffle(train_data)
            valid_size = int(len(train_data) * args.validation_percent)
            valid_data = train_data[-valid_size:]
            train_data = train_data[:-valid_size]
        if args.sst_file:
            train_data += load_dataset(args.sst_file, en_tweets=True if args.lang_id == "en" else False)
        train_dataset = AfriSentDataset(train_data, remove_handles_urls=args.remove_handles_urls)
        valid_dataset = AfriSentDataset(valid_data, remove_handles_urls=args.remove_handles_urls)
        partial_collate_fn = partial(collate_fn, tokenizer=tokenizer, max_length=args.tokenizer_max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=partial_collate_fn, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=2*args.batch_size, num_workers=1, collate_fn=partial_collate_fn)
        steps_per_epoch = math.ceil((len(train_dataset) / args.batch_size) / args.gradient_accumulations)
        if args.normalize_class_dist:
            class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=train_dataset.data_labels)
            print(class_weights)
        else:
            class_weights = [1, 1, 1]
        loss_fct_params = {"label_smoothing": args.label_smoothing, "weight": torch.tensor(class_weights).float()}
        model = Transformer(args.model_type, steps_per_epoch=steps_per_epoch, epochs=args.epochs, lr=args.lr, loss_fct_params=loss_fct_params)
        if args.checkpoint:
            model.load_state_dict(state_dict, strict=False)
        trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, default_root_dir="checkpoints", precision="bf16",
                            callbacks=[lr_callback, checkpoint_callback], accumulate_grad_batches=args.gradient_accumulations, 
                            val_check_interval=args.validation_interval)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    

if __name__ == "__main__":
    args = parse_training_args()
    train(args)