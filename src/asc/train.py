import os
import time
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel
from .model import AspectClassifier
from .data import SemDataset, collate_batch
from cfg import *

def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DATA_PATH,
                        help="Path or url of the dataset. If empty download from S3.")
    # parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default=MODEL_PATH,
                        help="Path, url or short name of the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    # TODO change with batch_size
    parser.add_argument("--acc_steps", type=int, default=8,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warm up steps")
    parser.add_argument("--valid_steps", type=int, default=2000, help="Perfom validation every X steps")
    # parser.add_argument("--n_emd", type=int, default=768, help="Number of n_emd in config file (for noam)")
    # parser.add_argument("--from_step", type=int, default=-1, help="Init learning rate from this step")
    args = parser.parse_args()

    log_dir_base = 'logs/asc_{}_{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())), log_mark)
    log_dir_train = log_dir_base + '_train'
    log_dir_dev = log_dir_base + '_dev'
    writer_train = SummaryWriter(os.path.join(MAIN_PATH, log_dir_train), flush_secs=5)
    writer_dev = SummaryWriter(os.path.join(MAIN_PATH, log_dir_dev), flush_secs=5)

    tokenizer = BertTokenizer.from_pretrained(args.model_checkpoint)
    transformer = BertModel.from_pretrained(args.model_checkpoint)
    model = AspectClassifier(transformer)

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
    # TODO warmup
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    # TODO param
    criterion = CrossEntropyLoss()

    train_dataset = SemDataset(tokenizer, args.dataset_path, 'train')
    dev_dataset = SemDataset(tokenizer, args.dataset_path, 'dev')
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_batch)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_batch)
    steps_per_epoch = len(train_dataloader)

    train_loss = []
    for epoch in range(args.n_epochs):
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, asp_masks, labels = batch
            output = model(input_ids, asp_masks)
            # TODO label smoothing
            loss = criterion(output, labels)
            loss = loss.mean() / args.acc_steps
            loss.backward()
            train_loss.append(loss.item())

            global_steps = epoch*steps_per_epoch + step + 1
            if global_steps % args.acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                train_loss_mean = sum(train_loss) / len(train_loss)
                writer_train.add_scalar('loss', train_loss_mean, global_steps)
                train_loss.clear()
                if global_steps % (args.acc_steps * 6) == 0:
                    print(f'  epoch {epoch}, step {step + 1}/{steps_per_epoch} loss: ', train_loss_mean)
            if global_steps % args.valid_steps == 1:
                model.eval()
                with torch.no_grad():
                    total = 0
                    correct = 0
                    dev_loss = 0.0
                    for batch in tqdm(dev_dataloader, position=0):
                        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
                        input_ids, asp_masks, labels = batch
                        output = model(input_ids, asp_masks)
                        loss = criterion(output, labels)
                        loss = loss.mean() / args.acc_steps
                        loss.backward()
                        dev_loss += loss.item()
                        _, pred = torch.max(output, dim=1)
                        total += labels.size(0)
                        correct += (pred == labels).sum().item()
                    dev_loss /= len(dev_dataloader)
                    writer_dev.add_scalar('loss', dev_loss, global_steps)
                    # TODO f1 score
                    writer_dev.add_scalar('acc', correct / total, global_steps)
