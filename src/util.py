import pandas as pd
import re


lang_ids = [
    "am",
    "dz",
    "ha",
    "ig",
    "kr",
    "ma",
    "pcm",
    "pt",
    "sw",
    "ts",
    "twi",
    "yo"
]



def load_dataset(file_path, en_tweets=False, just_text=False):
    dataset = []
    with open(file_path) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
                continue
            if not just_text:
                split = line.strip().split("\t", maxsplit=2)
                if len(split) == 3:
                    s1, s2, s3 = split
                elif len(split) == 2:
                    s1, s2, s3 = split + [None]
                else:
                    print("Issue, expected 2 or 3 spltis from splitting on tabs")
                    exit(1)
            else:
                s3 = line.strip() # we assume here that the only text only dataset will be "en_tweets = True"
                s1, s2 = None, None
            if not en_tweets:
                sample = {
                    "sample_id": s1,
                    "text": s2,
                    "label": s3
                }
            else:
                sample = {
                    "sample_id": s1,
                    "text": re.sub(r"@[^\s]*", "@user", s3),
                    "label": s2
                } 
            dataset.append(sample)
    return dataset


def to_cuda_hf(inputs):
    for k, v in inputs.items():
        inputs[k] = v.cuda()


def dump_predictions(tsv_file, output_file, predictions, keep_tweets=False):
    dev_pd = pd.read_csv(tsv_file, sep="\t")
    dev_pd["label"] = predictions
    if not keep_tweets:
        del dev_pd["tweet"]
    dev_pd.to_csv(output_file, sep="\t", index=False)
