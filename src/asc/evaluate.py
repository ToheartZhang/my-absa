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
from asc.criterion import LabelSmoothingLoss
from cfg import *
from utils import compute_f_score, save_model

def asc_evaluate(matched_ids=None):
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DATA_PATH,
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_name", type=str, default='laptop',
                        help="Dataset name.", choices=['restaurant', 'laptop'])
    # parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default=os.path.join(MODEL_PATH, 'asc_tokenizer'),
                        help="Path, url or short name of the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--num_classes", type=int, default=3, help="Num of classes for classification")
    parser.add_argument("--dropout", type=int, default=0.5, help="Rate of dropout")
    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(args.model_checkpoint)
    transformer = RobertaModel.from_pretrained('roberta-base', mirror='tuna')
    model = AspectClassifier(transformer, dropout=args.dropout)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, args.dataset_name + '_asc', '2021-05-14_14-52-26_0.8112966577507387.pt')), strict=True)
    # model.load_state_dict(torch.load(os.path.join(MODEL_PATH, args.dataset_name + '_asc', '2021-05-09_23-00-20_0.7798739261789357.pt')))
    model = model.to(args.device)

    test_dataset = SemDataset(tokenizer, args.dataset_path, args.dataset_name, 'test', matched_ids)
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
    asc_evaluate()

"""
restaurant roberta 2021-05-10_17-40-37_0.800588572778863.pt
f1: 0.776287003497838	precision: 0.8049256739732926	recall: 0.757326007326007	acc: 0.8491071428571428
f1: 0.7705125940078038	precision: 0.7910035888798994	recall: 0.7682496075353215	acc: 0.84375

restaurant robert-large 2021-05-11_16-12-58_0.8480032285070821.pt
f1: 0.7993358497922273	precision: 0.8457562565338215	recall: 0.7821166928309783	acc: 0.8732142857142857

laptop roberta 2021-05-09_22-41-19_0.823110779280058.pt
f1: 0.7577339596879217	precision: 0.7579559434153684	recall: 0.7740195727556144	acc: 0.7946708463949843
f1: 0.7278851950288568	precision: 0.7358893466131299	recall: 0.7318433608368468	acc: 0.774294670846395
f1: 0.7383419516785225	precision: 0.7478801559672137	recall: 0.7561563423652439	acc: 0.7821316614420063
f1: 0.7946883170496268	precision: 0.7977807665553144	recall: 0.8096100614129456	acc: 0.8307210031347962
f1: 0.7825643077811897	precision: 0.7808502208751803	recall: 0.8031305310694262	acc: 0.8197492163009404


laptop roberta-large 2021-05-11_14-50-15_0.8342186341060799.pt
f1: 0.7811823203468569	precision: 0.7806972737407515	recall: 0.7881199587736494	acc: 0.8119122257053292
"""
