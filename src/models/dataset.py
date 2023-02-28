import torch
import re
from transformers import AutoTokenizer
import torch.nn.functional as F

LANG_TO_TOKENIZER_ID = {
        "am": "amh_Ethi",
        "dz": "aeb_Arab",
        "ha": "hau_Latn",
        "ig": "ibo_Latn",
        "kr": "kin_Latn",
        "ma": "aeb_Arab",
        "pcm": "eng_Latn",
        "pt": "por_Latn",
        "sw": "swh_Latn",
        "ts": "tso_Latn",
        "twi": "twi_Latn",
        "yo": "yor_Latn",
        "or": "gaz_Latn",
        "tg": "tir_Ethi"
}


class AfriSentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, training=False, lang_id=None, remove_handles_urls=False):
        self.training = training
        self.label2id = {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }
        self.dataset = dataset
        self.lang_id = lang_id
        self.remove_handles_urls = remove_handles_urls
        self.data_labels = [self.label2id[sample["label"]] for sample in self.dataset if sample["label"] is not None]


    def clean_text(self, text):
        text = re.sub(r"@[^\s]*", " ", text)
        text = re.sub(r"http[^\s]*", " ", text)
        text = text.replace("  ", " ").replace("  ", " ").strip()
        return text


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample["text"]
        if self.remove_handles_urls:
            text = self.clean_text(text)
        if self.lang_id:
            text = f"<{self.lang_id}> " + text
        label_str = sample["label"]
        if label_str is None:
            return text, -100
        else:
            return text, self.label2id[label_str]



class NLLBAfriSentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, lang_id, model_type, training=False, remove_handles_urls=False, use_cls_token=False):
        self.training = training
        self.label2id = {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }
        self.dataset = dataset
        self.lang_id = lang_id
        self.remove_handles_urls = remove_handles_urls
        self.data_labels = [self.label2id[sample["label"]] for sample in self.dataset if sample["label"] is not None]
        self.use_cls_token = use_cls_token
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_type, src_lang=LANG_TO_TOKENIZER_ID[lang_id])
        except KeyError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_type, src_lang=lang_id)


    def clean_text(self, text):
        text = re.sub(r"@[^\s]*", " ", text)
        text = re.sub(r"http[^\s]*", " ", text)
        text = text.replace("  ", " ").replace("  ", " ").strip()
        return text


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample["text"]
        if self.remove_handles_urls:
            text = self.clean_text(text)
        if self.use_cls_token:
            text = self.tokenizer.bos_token + text
        label_str = sample["label"]
        if label_str is None:
            return self.tokenizer(text, return_tensors="pt"), -100
        else:
            return self.tokenizer(text, return_tensors="pt"), self.label2id[label_str]


def nllb_collate_fn(batch, tokenizer, max_length):
    batch_x, batch_y = [list(x) for x in zip(*batch)]
    max_seq_length = max(len(x["input_ids"][0]) for x in batch_x)
    padding_lengths = [max_seq_length - len(x["input_ids"][0]) for x in batch_x]
    batch_input_ids = []
    batch_attention_masks = []
    for inputs, padding_length in zip(batch_x, padding_lengths):
        batch_input_ids.append(F.pad(inputs["input_ids"][0], (0, padding_length), value=tokenizer.pad_token_id))
        batch_attention_masks.append(F.pad(inputs["attention_mask"][0], (0, padding_length), value=0))
    batch_x = {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_masks)
    }
    batch_y = torch.tensor(batch_y)
    return batch_x, batch_y


def collate_fn(batch, tokenizer, max_length):
    batch_x, batch_y = [list(x) for x in zip(*batch)]
    batch_x = tokenizer(batch_x, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    batch_y = torch.tensor(batch_y)
    return batch_x, batch_y


class MaskedLMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file, skip_samples=0, max_samples = float("inf"), training=False):
        self.training = training
        self.dataset_file = dataset_file
        self.dataset = []
        skipped = 0
        with open(self.dataset_file) as f:
            for line in f:
                if skipped < skip_samples:
                    skipped += 1
                    continue
                self.dataset.append(line.strip())
                if len(self.dataset) > max_samples:
                    break

    
    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample


# from: https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py
def torch_mask_tokens(inputs, tokenizer, special_tokens_mask = None, mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def masked_lm_collate_fn(batch, tokenizer, max_length, mlm_probability=0.15):
    batch_x = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    batch_x["input_ids"], batch_y = torch_mask_tokens(batch_x["input_ids"], tokenizer, mlm_probability=mlm_probability)
    return batch_x, batch_y