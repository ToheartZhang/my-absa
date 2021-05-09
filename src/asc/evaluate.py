import os
import sys
import time
import math
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer, RobertaModel
from asc.model import AspectClassifier
from asc.data import SemDataset, collate_batch
from asc.optimizer import LabelSmoothingLoss
from cfg import *
from utils import compute_f_score, save_model

def evaluate():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DATA_PATH,
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_name", type=str, default='laptop',
                        help="Dataset name.", choices=['restaurant', 'laptop'])
    # parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default=MODEL_PATH,
                        help="Path, url or short name of the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--num_classes", type=int, default=3, help="Num of classes for classification")
    parser.add_argument("--dropout", type=int, default=0.1, help="Rate of dropout")
    parser.add_argument("--smoothing", type=int, default=0.0, help="Rate of label smoothing")
    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(args.model_checkpoint)
    transformer = RobertaModel.from_pretrained('roberta-base', mirror='tuna')
    model = AspectClassifier(transformer, dropout=args.dropout)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, args.dataset_name + '_asc', '2021-05-09_22-41-19_0.823110779280058.pt')))
    # model.load_state_dict(torch.load(os.path.join(MODEL_PATH, args.dataset_name + '_asc', '2021-05-09_23-00-20_0.7798739261789357.pt')))
    model = model.to(args.device)

    test_dataset = SemDataset(tokenizer, args.dataset_path, args.dataset_name, 'test')
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_batch)

    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        tps, fps, fns = [0 for _ in range(args.num_classes)], [0 for _ in range(args.num_classes)], [0 for _ in range(
            args.num_classes)]
        for batch in tqdm(test_dataloader, position=0):
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, asp_masks, labels = batch
            output = model(input_ids, asp_masks)
            _, pred = torch.max(output, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            for idx in range(args.num_classes):
                tps[idx] += torch.sum((pred == idx).long().masked_fill(labels != idx, 0)).item()
                fps[idx] += torch.sum((pred == idx).long().masked_fill(labels == idx, 0)).item()
                fns[idx] += torch.sum((pred != idx).long().masked_fill(labels != idx, 0)).item()
        f_sum, pre_sum, rec_sum = 0.0, 0.0, 0.0
        for idx in range(args.num_classes):
            f, pre, rec = compute_f_score(tps[idx], fns[idx], fps[idx])
            f_sum += f
            pre_sum += pre
            rec_sum += rec
        f_score = f_sum / args.num_classes
        pre = pre_sum / args.num_classes
        rec = rec_sum / args.num_classes
        print(f'f1: {f_score}\tprecision: {pre}\trecall: {rec}\tacc: {correct/total}')

if __name__ == '__main__':
    evaluate()

"""
2021-05-09_23-00-20_0.7798739261789357.pt
f1: 0.8135673667588045	precision: 0.9375150204277715	recall: 0.7765795206971643	acc: 0.9166666666666666

2021-05-09_22-41-19_0.823110779280058.pt
f1: 0.8783068783068181	precision: 0.9509803921568479	recall: 0.8333333333333245	acc: 0.8979591836734694
"""
