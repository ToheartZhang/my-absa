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

def collate_batch(batch):
    batch_token_ids = []
    batch_attn_masks = []
    batch_token_type_ids = []
    batch_labels = []
    batch_seq_lens = []
    for sample in batch:
        token_ids, attn_mask, token_type_ids, label, seq_lens = sample
        batch_token_ids.append(token_ids)
        batch_attn_masks.append(attn_mask)
        batch_token_type_ids.append(token_type_ids)
        batch_labels.append(label)
        batch_seq_lens.append(seq_lens)

    return torch.tensor(batch_token_ids, dtype=torch.long), \
           torch.tensor(batch_attn_masks, dtype=torch.long), \
           torch.tensor(batch_token_type_ids, dtype=torch.long), \
           torch.tensor(batch_labels, dtype=torch.long), \
           batch_seq_lens

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
        print(f'construct dataset from {data_path}')
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            cnt = 0
            for line in tqdm(f.readlines()):
                line = line.strip()
                line = json.loads(line)
                tokens = []
                label_ids = []
                text = line['text'].copy()
                labels = line['label'].copy()
                for idx, (word, label) in enumerate(zip(text, labels)):
                    if idx == 0:
                        word_tokens = tokenizer.tokenize(word)
                    else:
                        word_tokens = tokenizer.tokenize(' ' + word)
                    tokens.extend(word_tokens)
                    label_ids.extend([label_dict[label]] + [ignore_index] * (len(word_tokens) - 1))

                special_tokens_count = 3
                if len(tokens) > max_seq_len - special_tokens_count:
                    tokens = tokens[:(max_seq_len - special_tokens_count)]
                    label_ids = label_ids[:(max_seq_len - special_tokens_count)]

                tokens = [cls] + tokens + [sep]
                label_ids = [ignore_index] + label_ids + [ignore_index]
                token_type_ids = [0] * len(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                seq_lens = len(input_ids)

                pad_length = max_seq_len - len(input_ids)
                input_ids += ([pad_id] * pad_length)
                label_ids += ([0] * pad_length)
                token_type_ids += ([0] * pad_length)
                input_mask += ([0] * pad_length)

                assert len(input_ids) == max_seq_len
                assert len(label_ids) == max_seq_len
                assert len(token_type_ids) == max_seq_len
                assert len(input_mask) == max_seq_len

                # input_ids = torch.tensor(input_ids, dtype=torch.long)
                # attention_mask = torch.tensor(input_mask, dtype=torch.long)
                # token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
                # label = torch.tensor(label_ids, dtype=torch.long)

                if cnt < 5:
                    print("*** Example ***")
                    print("tokens:", tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens)))
                    print("input_ids:", " ".join([str(x) for x in input_ids]))
                    print("input_mask:", " ".join([str(x) for x in input_mask]))
                    print("segment_ids:", " ".join([str(x) for x in token_type_ids]))
                    print("label_ids:", " ".join([str(x) for x in label_ids]))
                cnt += 1
                self.data.append((input_ids, input_mask, token_type_ids, label_ids, seq_lens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
