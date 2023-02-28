from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.models.transformer import ECELoss, Transformer
from torch.utils.data import DataLoader
import sys
import torch
from src.util import to_cuda_hf

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

from tqdm import tqdm

model = Transformer("xlm-roberta-large")
model.cuda()
state_dict = torch.load(sys.argv[1], map_location=torch.device("cpu"))["state_dict"]
if "model.loss_fct.weight" in state_dict:
    del state_dict["model.loss_fct.weight"]
model.load_state_dict(state_dict)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
valid_data = load_dataset("SubtaskA/dev/en_dev.tsv", en_tweets=True)
valid_dataset = AfriSentDataset(valid_data)
partial_collate_fn = partial(collate_fn, tokenizer=tokenizer, max_length=128)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, collate_fn=partial_collate_fn, drop_last=True)
labels = []
logits = []
with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_x, batch_y in tqdm(valid_dataloader):
            to_cuda_hf(batch_x)
            outputs = model(batch_x)
            labels.append(batch_y.detach().cpu())
            logits.append(outputs.logits.detach().cpu())

ece = ECELoss()
print(ece(torch.vstack(logits), torch.stack(labels).reshape(-1)))