import os
import sys
import shutil
import time
import math
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer, RobertaForTokenClassification, RobertaConfig, WEIGHTS_NAME
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from ate.data import SemAEDataset, label_dict, collate_batch
from cfg import *
from utils import compute_f_score, save_model

MODEL_CLASS = {
    'roberta': (RobertaForTokenClassification, RobertaTokenizer, RobertaConfig)
}


def get_ate_f_score(y_preds, y_trues, b=1):
    common, relevant, retrieved = 0., 0., 0.
    pred_terms = []
    true_terms = []
    for y_pred, y_true in zip(y_preds, y_trues):
        idx = 0
        sample_pred_terms = []
        sample_true_terms = []
        while idx < len(y_pred):
            if y_pred[idx] == 1:
                start = idx
                idx += 1
                while idx < len(y_pred) and y_pred[idx] == 2:
                    idx += 1
                end = idx
                sample_pred_terms.append((start, end))
            else:
                idx += 1
        idx = 0
        while idx < len(y_true):
            if y_true[idx] == 1:
                start = idx
                idx += 1
                while idx < len(y_true) and y_true[idx] == 2:
                    idx += 1
                end = idx
                sample_true_terms.append((start, end))
            else:
                idx += 1
        common += len([a for a in sample_pred_terms if a in sample_true_terms])
        retrieved += len(sample_pred_terms)
        relevant += len(sample_true_terms)
        pred_terms.append(sample_pred_terms)
        true_terms.append(sample_true_terms)
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (1 + (b ** 2)) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.

    cnt = 0
    matched_ids = []
    for s_true_terms, s_pred_terms in zip(true_terms, pred_terms):
        for t in s_true_terms:
            if t in s_pred_terms:
                matched_ids.append(cnt)
            cnt += 1
    print("Total true terms: ", cnt)
    return p, r, f1, common, retrieved, relevant, matched_ids


def ate_evaluate():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DATA_PATH,
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_name", type=str, default='laptop',
                        help="Dataset name.", choices=['restaurant', 'laptop'])
    parser.add_argument("--model_name", type=str, default='roberta',
                        help="Model name")
    parser.add_argument("--pretrain_name", type=str, default='roberta-base',
                        help="Path, url or short name of the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--do_lower_case", type=bool, default=False)
    args = parser.parse_args()

    transformer_class, tokenizer_class, config_class = MODEL_CLASS[args.model_name]
    config = config_class.from_pretrained(args.pretrain_name, num_labels=3)
    tokenizer = tokenizer_class.from_pretrained(os.path.join(MODEL_PATH, 'ate_tokenizer'),
                                                do_lower_case=args.do_lower_case)
    model = transformer_class.from_pretrained(args.pretrain_name, config=config, mirror='tuna')
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, args.dataset_name + '_ate', '2021-05-14_23-18-51_0.642606001506814.pt')))
    model = model.to(args.device)
    model.eval()

    test_dataset = SemAEDataset(tokenizer, args.dataset_path, args.dataset_name, 'test')
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_batch)

    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(test_dataloader)):
        # batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input = {
            "input_ids": batch[0].to(args.device),
            "attention_mask": batch[1].to(args.device),
            "token_type_ids": batch[2].to(args.device),
            "labels": batch[3].to(args.device)
        }
        labels, seq_lens = batch[3:5]
        output = model(**input)
        _, logits = output[:2]
        _, pred = torch.max(logits, dim=2)
        for i in range(pred.size(0)):
            sample_y_true = []
            sample_y_pred = []
            for j in range(seq_lens[i]):
                if labels[i, j] == -100:
                    continue
                sample_y_true.append(labels[i, j].item())
                sample_y_pred.append(pred[i, j].item())
            y_pred.append(sample_y_pred)
            y_true.append(sample_y_true)
    p, r, f1, common, retrieved, relevant, matched_ids = get_ate_f_score(y_pred, y_true)
    print(f'f1: {f1}\tprecision: {p}\trecall: {r}\tacc: {common / retrieved}')
    return matched_ids


if __name__ == '__main__':
    ate_evaluate()

"""
laptop
f1: 0.8357723577235773	precision: 0.8624161073825504	recall: 0.8107255520504731	acc: 0.8624161073825504
"""