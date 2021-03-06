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
from sklearn.metrics import classification_report
from ate.data import SemAEDataset, collate_batch
from cfg import *
from utils import compute_f_score, save_model

MODEL_CLASS = {
    'roberta': (RobertaForTokenClassification, RobertaTokenizer, RobertaConfig)
}

def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DATA_PATH,
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_name", type=str, default='laptop',
                        help="Dataset name.", choices=['restaurant', 'laptop'])
    parser.add_argument("--model_name", type=str, default='roberta',
                        help="Model name")
    parser.add_argument("--model_checkpoint", type=str, default='roberta-base',
                        help="Path, url or short name of the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warm up steps")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--do_lower_case", type=bool, default=False)
    parser.add_argument("--valid_steps", type=int, default=50, help="Perfom validation every X steps")
    parser.add_argument("--num_classes", type=int, default=3, help="Num of classes for classification")
    args = parser.parse_args()

    log_mark = f'{args.batch_size}_{args.lr}'
    log_dir_base = 'logs/ate_{}_{}_{}'.format(args.dataset_name, time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())), log_mark)
    log_dir_train = log_dir_base + '_train'
    log_dir_dev = log_dir_base + '_dev'
    writer_train = SummaryWriter(os.path.join(MAIN_PATH, log_dir_train), flush_secs=5)
    writer_dev = SummaryWriter(os.path.join(MAIN_PATH, log_dir_dev), flush_secs=5)

    transformer_class, tokenizer_class, config_class = MODEL_CLASS[args.model_name]
    config = config_class.from_pretrained(args.model_checkpoint, num_labels=3)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, do_lower_case=args.do_lower_case)
    tokenizer.save_pretrained(os.path.join(MODEL_PATH, 'ate_tokenizer'))
    model = transformer_class.from_pretrained(args.model_checkpoint, mirror='tuna', config=config)
    model = model.to(args.device)

    train_dataset = SemAEDataset(tokenizer, args.dataset_path, args.dataset_name, 'train')
    dev_dataset = SemAEDataset(tokenizer, args.dataset_path, args.dataset_name, 'dev')
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_batch)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_batch)
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch*args.n_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    train_loss = []
    best_f_score = 0.0
    for epoch in range(args.n_epochs):
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch[:-1])
            input = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]
            }
            output = model(**input)
            loss = output[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            train_loss.append(loss.item())

            global_steps = epoch*steps_per_epoch + step + 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss_mean = sum(train_loss) / len(train_loss)
            writer_train.add_scalar('loss', train_loss_mean, global_steps)
            writer_train.add_scalar('lr', scheduler.get_last_lr()[0], global_steps)
            train_loss.clear()
            if global_steps % 6 == 0:
                print(f'  epoch {epoch}, step {step + 1}/{steps_per_epoch} loss: ', train_loss_mean)
            if global_steps % args.valid_steps == 1:
                model.eval()
                with torch.no_grad():
                    dev_loss = 0.0
                    total_tokens = 0
                    y_true = []
                    y_pred = []
                    for batch in tqdm(dev_dataloader, position=0):
                        # batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
                        input = {
                            "input_ids": batch[0].to(args.device),
                            "attention_mask": batch[1].to(args.device),
                            "token_type_ids": batch[2].to(args.device),
                            "labels": batch[3].to(args.device)
                        }
                        labels, seq_lens = batch[3:5]
                        output = model(**input)
                        loss, logits = output[:2]
                        _, pred = torch.max(logits, dim=2)
                        for i in range(pred.size(0)):
                            for j in range(seq_lens[i]):
                                if labels[i, j] == -100 or (labels[i, j] == 0 and pred[i, j] == 0):
                                    continue
                                y_true.append(labels[i, j].item())
                                y_pred.append(pred[i, j].item())
                        dev_loss += loss.item()
                        total_tokens += sum(seq_lens)
                    dev_loss /= len(dev_dataloader)
                    report = classification_report(y_true, y_pred, output_dict=True)
                    print(classification_report(y_true, y_pred))
                    macro_precision = report['weighted avg']['precision']
                    macro_recall = report['weighted avg']['recall']
                    macro_f1 = report['weighted avg']['f1-score']
                    accuracy = report['accuracy']
                    writer_dev.add_scalar('loss', dev_loss, global_steps)
                    writer_dev.add_scalar('acc', accuracy, global_steps)
                    writer_dev.add_scalar('f1', macro_f1, global_steps)
                    writer_dev.add_scalar('pre', macro_precision, global_steps)
                    writer_dev.add_scalar('rec', macro_recall, global_steps)
                    print(f'  dev_loss {dev_loss}, f1 {macro_f1}, acc {accuracy}')

                    if macro_f1 > best_f_score and macro_f1 > 0.63:
                        best_f_score = macro_f1
                        save_model(model, path_prefix=f'{args.dataset_name}_ate', score=best_f_score)

if __name__ == '__main__':
    train()
