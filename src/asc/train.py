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
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer, RobertaModel, WEIGHTS_NAME
from asc.model import AspectClassifier
from asc.data import SemDataset, collate_batch
from asc.optimizer import LabelSmoothingLoss
from cfg import *
from utils import compute_f_score, save_model

def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DATA_PATH,
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_name", type=str, default='restaurant',
                        help="Dataset name.", choices=['restaurant', 'laptop'])
    # parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default='roberta-base',
                        help="Path, url or short name of the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--acc_batch_size", type=int, default=32,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--warmup_rate", type=int, default=0.01, help="Warm up steps")
    parser.add_argument("--valid_steps", type=int, default=50, help="Perfom validation every X steps")
    parser.add_argument("--num_classes", type=int, default=3, help="Num of classes for classification")
    parser.add_argument("--dropout", type=int, default=0.5, help="Rate of dropout")
    parser.add_argument("--smoothing", type=int, default=0.0, help="Rate of label smoothing")
    args = parser.parse_args()
    acc_steps = args.acc_batch_size // args.batch_size

    log_mark = f'{args.batch_size}_{args.lr}'
    log_dir_base = 'logs/asc_{}_{}_{}'.format(args.dataset_name, time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())), log_mark)
    log_dir_train = log_dir_base + '_train'
    log_dir_dev = log_dir_base + '_dev'
    writer_train = SummaryWriter(os.path.join(MAIN_PATH, log_dir_train), flush_secs=5)
    writer_dev = SummaryWriter(os.path.join(MAIN_PATH, log_dir_dev), flush_secs=5)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.save_pretrained(MODEL_PATH)
    transformer = RobertaModel.from_pretrained(args.model_checkpoint, mirror='tuna')
    model = AspectClassifier(transformer, dropout=args.dropout)
    model = model.to(args.device)

    train_dataset = SemDataset(tokenizer, args.dataset_path, args.dataset_name, 'train')
    dev_dataset = SemDataset(tokenizer, args.dataset_path, args.dataset_name, 'dev')
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_batch)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_batch)
    steps_per_epoch = len(train_dataloader)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-2,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    warmup_steps = int(args.n_epochs * steps_per_epoch * args.warmup_rate)
    linear_lambda = lambda epoch: (0.9 * epoch / warmup_steps + 0.1) if epoch < warmup_steps else 1
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lambda)
    # criterion = LabelSmoothingLoss(args.num_classes, smoothing=args.smoothing)
    criterion = CrossEntropyLoss()

    train_loss = []
    best_f_score = 0.0
    for epoch in range(args.n_epochs):
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, asp_masks, labels = batch
            output = model(input_ids, asp_masks)
            loss = criterion(output, labels)
            loss = loss.mean() / acc_steps
            if loss.item() > 10:
                print('===========================')
            loss.backward()
            train_loss.append(loss.item())

            global_steps = epoch*steps_per_epoch + step + 1
            if global_steps % acc_steps == 0:
                optimizer.step()
                scheduler.step(global_steps)
                optimizer.zero_grad()
                train_loss_mean = sum(train_loss) / len(train_loss)
                writer_train.add_scalar('loss', train_loss_mean, global_steps)
                train_loss.clear()
                if global_steps % (acc_steps * 6) == 0:
                    print(f'  epoch {epoch}, step {step + 1}/{steps_per_epoch} loss: ', train_loss_mean)
            if global_steps % args.valid_steps == 1:
                model.eval()
                with torch.no_grad():
                    total = 0
                    correct = 0
                    dev_loss = 0.0
                    tps, fps, fns = [0 for _ in range(args.num_classes)], [0 for _ in range(args.num_classes)], [0 for _ in range(args.num_classes)]
                    for batch in tqdm(dev_dataloader, position=0):
                        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
                        input_ids, asp_masks, labels = batch
                        output = model(input_ids, asp_masks)
                        loss = criterion(output, labels)
                        loss = loss.mean() / acc_steps
                        dev_loss += loss.item()
                        _, pred = torch.max(output, dim=1)
                        total += labels.size(0)
                        correct += (pred == labels).sum().item()
                        for idx in range(args.num_classes):
                            tps[idx] += torch.sum((pred == idx).long().masked_fill(labels != idx, 0)).item()
                            fps[idx] += torch.sum((pred == idx).long().masked_fill(labels == idx, 0)).item()
                            fns[idx] += torch.sum((pred != idx).long().masked_fill(labels != idx, 0)).item()
                    dev_loss /= len(dev_dataloader)
                    writer_dev.add_scalar('loss', dev_loss, global_steps)
                    f_sum, pre_sum, rec_sum = 0.0, 0.0, 0.0
                    for idx in range(args.num_classes):
                        f, pre, rec = compute_f_score(tps[idx], fns[idx], fps[idx])
                        f_sum += f
                        pre_sum += pre
                        rec_sum += rec
                    f_score = f_sum / args.num_classes
                    writer_dev.add_scalar('acc', correct / total, global_steps)
                    writer_dev.add_scalar('f1', f_score, global_steps)
                    writer_dev.add_scalar('pre', pre_sum / args.num_classes, global_steps)
                    writer_dev.add_scalar('rec', rec_sum / args.num_classes, global_steps)
                    if f_score > best_f_score:
                        best_f_score = f_score
                        save_model(model, path_prefix=f'{args.dataset_name}_asc', score=best_f_score)

if __name__ == '__main__':
    train()
