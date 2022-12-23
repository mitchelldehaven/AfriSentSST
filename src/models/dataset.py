import torch


class AfriSentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, training=False):
        self.training = training
        self.label2id = {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }
        self.dataset = dataset


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample["text"]
        label_str = sample["label"]
        if label_str is None:
            return text, -100
        else:
            return text, self.label2id[label_str]


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