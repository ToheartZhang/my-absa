import os
import pickle
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class SemDataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type):
        super(Dataset, self).__init__()
        cls = tokenizer.cls_token
        sep = tokenizer.sep_token
        data_path = os.path.join(data_dir, f'{data_type}.txt')
        cache_path = os.path.join(data_dir, f'{data_type}_cache.pkl')
        if os.path.isfile(cache_path):
            self.data = load_pkl(cache_path)
        else:
            self.data = ()
            all_tokens = []
            all_asp_masks = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                    line = line.strip()
                    line = json.loads(line)

                    sentence = line['sentence']
                    aspects = line['aspects']
                    tokens = tokenizer.convert_tokens_to_ids(tokenizer.encode(sentence))
                    tokens = [cls] + tokens + [sep]
                    starts = []
                    ends = []
                    for asp in aspects:
                        st, ed = asp[2], asp[3]
                        starts.append(st + 1)   # for cls
                        ends.append(ed + 1)
                        asp_mask = [0] * len(tokens)
                        for idx in range(st, ed):
                            asp_mask[idx] = 1

                        all_tokens.append(tokens.copy())
                        all_asp_masks.append(asp_mask)
            self.data = (all_tokens, all_asp_masks)
            save_pkl(self.data, cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]
