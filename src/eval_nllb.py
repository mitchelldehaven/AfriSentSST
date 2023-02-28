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
from src.models import Transformer, NLLB_Encoder
from src.models.dataset import AfriSentDataset, NLLBAfriSentDataset, collate_fn, nllb_collate_fn
import src.util as utils
import pandas as pd
from tqdm import tqdm


# def init_lang_tokens(tokenizer):
#     for lang_id in utils.lang_ids:
#             tokenizer.add_tokens(f"<{lang_id}>")


def parse_inference_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_id", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="dev")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tokenizer_max_length", type=int, default=64)
    parser.add_argument("--model_type", type=str, default="facebook/nllb-200-distilled-1.3B")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--remove_handles_urls", action="store_true")
    parser.add_argument("--use_cls_token", action="store_true")
    parser.add_argument("--subtask_dir", type=str, default="SubtaskA")
    parser.add_argument("--keep_tweets", action="store_true")
    args = parser.parse_args()
    return args


def evaluate(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    # if args.multi:
    #     init_lang_tokens(tokenizer)
    model = NLLB_Encoder(args.model_type, tokenizer)
    state_dict = torch.load(args.checkpoint)["state_dict"]
    if "model.loss_fct.weight" in state_dict:
        del state_dict["model.loss_fct.weight"]
    model.load_state_dict(state_dict)
    model.cuda()
    if args.dataset == "dev":
        filename_suffix = "dev"
    else:
        filename_suffix = "test_participants"
    dev_file = ROOT_DIR / args.subtask_dir / args.dataset / f"{args.lang_id}_{filename_suffix}.tsv"
    args.output_dir.mkdir(exist_ok=True, parents=True)
    output_file = args.output_dir / f"pred_{args.lang_id}.tsv"
    dev_pd = pd.read_csv(dev_file, sep="\t")
    dev_data = load_dataset(dev_file)
    dev_dataset = NLLBAfriSentDataset(dev_data, args.lang_id, args.model_type, remove_handles_urls=True, use_cls_token=args.use_cls_token)
    id2label = {v: k for k, v in dev_dataset.label2id.items()}
    partial_collate_fn = partial(nllb_collate_fn, tokenizer=tokenizer, max_length=args.tokenizer_max_length)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=partial_collate_fn)
    predictions = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_inputs, batch_labels in tqdm(dev_dataloader):
            to_cuda_hf(batch_inputs)
            batch_outputs = model(batch_inputs)
            batch_predictions = batch_outputs[0].argmax(axis=1).detach().cpu()
            for pred in batch_predictions:
                pred_str = id2label[pred.item()]
                predictions.append(pred_str)
    
    dump_predictions(dev_file, output_file, predictions, keep_tweets=args.keep_tweets)


if __name__ == "__main__":
    args = parse_inference_args()
    evaluate(args) 