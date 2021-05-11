import os
import pickle
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

label_dict = {'O': 0, 'B': 1, 'I': 2}

def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class SemAEDataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_name, data_type, ignore_index=-100, max_seq_len=128):
        super(Dataset, self).__init__()
        cls = tokenizer.cls_token
        sep = tokenizer.sep_token
        pad = tokenizer.pad_token
        # cls_id = tokenizer.convert_tokens_to_ids(cls)
        # sep_id = tokenizer.convert_tokens_to_ids(sep)
        pad_id = tokenizer.convert_tokens_to_ids(pad)
        data_path = os.path.join(data_dir, data_name, f'{data_type}.json')
        cache_path = os.path.join(data_dir, data_name, f'{data_type}_cache.pkl')
        if os.path.isfile(cache_path):
            print(f'load dataset from {cache_path}')
            self.data = load_pkl(cache_path)
        else:
            print(f'construct dataset from {data_path}')
            self.data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(tqdm(f.readlines())):
                    line = line.strip()
                    line = json.loads(line)

                    tokens = []
                    label_ids = []
                    text = line['sentence']
                    labels = line['label']
                    for word, label in zip(text, labels):
                        word_tokens = tokenizer.tokenize(word)
                        tokens.extend(word_tokens)
                        label_ids.extend([label_dict[label]] + [ignore_index]*(len(word_tokens - 1)))
                    # TODO max_seq_len
                    tokens = [tokenizer.tokenize(cls)] + tokens + [tokenizer.tokenize(sep)]
                    label_ids = [ignore_index] + tokens + [ignore_index]
                    token_type_ids = [0] * len(tokens)
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    input_mask = [1]*len(input_ids)

                    pad_length = max_seq_len - len(input_ids)
                    input_ids += ([pad_id]*pad_length)
                    label_ids += ([0]*pad_length)
                    token_type_ids += ([0]*pad_length)
                    input_mask += ([0]*pad_length)

                    assert len(input_ids) == max_seq_len
                    assert len(label_ids) == max_seq_len
                    assert len(token_type_ids) == max_seq_len
                    assert len(input_mask) == max_seq_len

                    if idx < 5:
                        print("*** Example ***")
                        print("tokens: %s", " ".join([str(x) for x in tokens]))
                        print("input_ids: %s", " ".join([str(x) for x in input_ids]))
                        print("input_mask: %s", " ".join([str(x) for x in input_mask]))
                        print("segment_ids: %s", " ".join([str(x) for x in token_type_ids]))
                        print("label_ids: %s", " ".join([str(x) for x in label_ids]))

                    self.data.append((input_ids, input_mask, token_type_ids, label_ids))
            save_pkl(self.data, cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
