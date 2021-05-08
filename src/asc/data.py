import os
import pickle
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def padded_tensor(items, pad_idx=0, max_len=None, pad_tail=True):
    n = len(items)
    lens = [len(item) for item in items]
    t = max(lens) if max_len is None else max_len
    t = max(t, 1)
    if isinstance(items[0], torch.Tensor):
        # keep type of input tensors, they may already be cuda ones
        output = items[0].new(n, t)  # type: ignore
    else:
        output = torch.LongTensor(n, t)  # type: ignore
    output.fill_(pad_idx)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            # skip empty items
            continue
        if not isinstance(item, torch.Tensor):
            # put non-tensors into a tensor
            item = torch.tensor(item, dtype=torch.long)  # type: ignore
        if pad_tail:
            # place at beginning
            output[i, :length] = item
        else:
            # place at end
            output[i, t - length:] = item

    return output

def collate_batch(batch):
    batch_token_ids = []
    batch_asp_masks = []
    batch_labels = []
    for sample in batch:
        token_ids, asp_masks, labels = sample
        batch_token_ids.append(token_ids)
        batch_asp_masks.append(asp_masks)
        batch_labels.append(labels)

    batch_token_ids = padded_tensor(batch_token_ids)
    batch_asp_masks = padded_tensor(batch_asp_masks)
    return batch_token_ids, batch_asp_masks, torch.Tensor(batch_labels)

class SemDataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type):
        super(Dataset, self).__init__()
        cls = tokenizer.cls_token
        sep = tokenizer.sep_token
        data_path = os.path.join(data_dir, f'{data_type}.json')
        cache_path = os.path.join(data_dir, f'{data_type}_cache.pkl')
        if os.path.isfile(cache_path):
            self.data = load_pkl(cache_path)
        else:
            self.data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                    line = line.strip()
                    line = json.loads(line)

                    text = line['text']
                    aspect = line['aspect']
                    tokens = tokenizer.convert_tokens_to_ids(tokenizer.encode(text))
                    tokens = [cls] + tokens + [sep]
                    start, end = aspect[2], aspect[3]
                    asp_mask = [0] * len(tokens)
                    for idx in range(start, end):
                        asp_mask[idx] = 1

                    self.data.append((tokens, asp_mask, aspect[1]))
            save_pkl(self.data, cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
