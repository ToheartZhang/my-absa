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

    batch_token_ids = padded_tensor(batch_token_ids, pad_idx=1) # TODO fix hard encode
    batch_asp_masks = padded_tensor(batch_asp_masks, pad_idx=0)
    return batch_token_ids, batch_asp_masks, torch.tensor(batch_labels)

class SemDataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_name, data_type, matched_ids=None):
        super(Dataset, self).__init__()
        cls = tokenizer.cls_token
        sep = tokenizer.sep_token
        data_path = os.path.join(data_dir, data_name, f'{data_type}.json')
        print(f'construct dataset from {data_path}')
        self.data = []
        cnt = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                line = json.loads(line)

                aspects = line['aspects'].copy()
                for aspect in aspects:
                    if aspect == None:
                        continue
                    if matched_ids is not None and cnt not in matched_ids:
                        cnt += 1
                        continue
                    cnt += 1
                    text = line['text'].copy()
                    text = [cls] + text + [sep]
                    start, end = aspect[2] + 1, aspect[3] + 1
                    asp_mask = [0] * len(text)
                    for idx in range(start, end):
                        asp_mask[idx] = 1
                    pieces = []
                    piece_masks = []
                    for idx, (mask, token) in enumerate(zip(asp_mask, text)):
                        if idx < 2 or idx == len(text) - 1:
                            bpes = tokenizer.convert_tokens_to_ids(
                                tokenizer.tokenize(token)
                            )
                        else:
                            bpes = tokenizer.convert_tokens_to_ids(
                                tokenizer.tokenize(' ' + token)
                            )
                        pieces.extend(bpes)
                        piece_masks.extend([mask]*len(bpes))
                    assert len(pieces) == len(piece_masks)
                    self.data.append((pieces, piece_masks, aspect[1]))
        print(f'Total true terms: {cnt}')
        print(f'Total match terms: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
