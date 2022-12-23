import pandas as pd


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


def load_dataset(file_path):
    dataset = []
    with open(file_path) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
                continue
            split = line.strip().split("\t")
            if len(split) == 3:
                sample_id, text, label = split
            else:
                sample_id, text, label = split + [None]
            sample = {
                "sample_id": sample_id,
                "text": text,
                "label": label
            }
            dataset.append(sample)
    return dataset


def to_cuda_hf(inputs):
    for k, v in inputs.items():
        inputs[k] = v.cuda()


def dump_predictions(tsv_file, output_file, predictions):
    dev_pd = pd.read_csv(tsv_file, sep="\t")
    dev_pd["label"] = predictions
    del dev_pd["tweet"]
    dev_pd.to_csv(output_file, sep="\t", index=False)