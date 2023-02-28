from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse
from src.util import load_dataset, to_cuda_hf
from src.paths import DATA_DIR
from src.models.dataset import AfriSentDataset, collate_fn
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from functools import partial
import pandas as pd
from tqdm import tqdm

TARGETS = [
    "amh_Ethi",
    "aeb_Arab",
    "hau_Latn",
    "ibo_Latn",
    "kin_Latn",
    "por_Latn",
    "swh_Latn",
    "tso_Latn",
    "twi_Latn",
    "yor_Latn",
    "tir_Ethi",
    "gaz_Latn"
]



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=Path, required=True)
    parser.add_argument("--model_type", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max_samples", type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # Assumes source language that is being translated is English
    args.output_dir.mkdir(exist_ok=True, parents=True)
    data = load_dataset(args.data_file, en_tweets=True, just_text=False)
    if args.max_samples:
        data = data[:args.max_samples]
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_type)
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, src_lang="eng_Latn")
    dataset = AfriSentDataset(data, remove_handles_urls=True)
    id2label = {v: k for k, v in dataset.label2id.items()}
    partial_collate_fn = partial(collate_fn, tokenizer=tokenizer, max_length=128)
    dataloader = DataLoader(dataset, num_workers=2, batch_size=256, collate_fn=partial_collate_fn, shuffle=False)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            for target in TARGETS:
                translations = []
                labels = []
                for batch_x, batch_y in tqdm(dataloader):
                    to_cuda_hf(batch_x)
                    translated_tokens_ids = model.generate(
                        **batch_x, forced_bos_token_id=tokenizer.lang_code_to_id[target], max_length=32
                    )
                    translated_tokens = tokenizer.batch_decode(translated_tokens_ids, skip_special_tokens=True)
                    batch_labels = [id2label[int(y)] for y in batch_y]
                    for translation, label in zip(translated_tokens, batch_labels):
                        translations.append(translation)
                        labels.append(label)
                df = pd.DataFrame.from_dict({"translations": translations, "labels": labels})
                df.to_csv(args.output_dir / f"en__{target}_pseudo_labelled.tsv", sep="\t")
