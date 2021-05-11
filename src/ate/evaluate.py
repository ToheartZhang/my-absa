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
from ate.data import SemAEDataset, label_dict
from cfg import *
from utils import compute_f_score, save_model

MODEL_CLASS = {
    'roberta': (RobertaForTokenClassification, RobertaTokenizer, RobertaConfig)
}

def evalute():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DATA_PATH,
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_name", type=str, default='ate_restaurant',
                        help="Dataset name.", choices=['ate_restaurant', 'ate_laptop'])
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
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--do_lower_case", type=bool, default=True)
    args = parser.parse_args()

    transformer_class, tokenizer_class, config_class = MODEL_CLASS[args.model_name]
    config = config_class.from_pretrained(args.model_checkpoint, num_labels=3)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, do_lower_case=args.do_lower_case)
    tokenizer.save_pretrained(os.path.join(MODEL_PATH, args.dataset_name + 'ate'))
    model = transformer_class.from_pretrained(args.model_checkpoint, mirror='tuna', config=config)
    model = model.to(args.device)
    model.eval()

    test_dataset = SemAEDataset(tokenizer, args.dataset_path, args.dataset_name, 'test')
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    preds = None
    out_label_ids = None
    for step, batch in enumerate(tqdm(test_dataloader)):
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3]
        }
        output = model(**input)
        _, logits = output[:2]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = input["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy())
            out_label_ids = np.append(out_label_ids, input["labels"].detach().cpu().numpy())
    preds = np.argmax(preds, axis=2)
    label_dict_re = {v: k for k, v in label_dict.items()}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100:
                out_label_list[i].append(label_dict_re[out_label_ids[i, j]])
                preds_list[i].append(label_dict_re[preds[i, j]])

            if i < 5:
                print("*** Evaluation Example ***")
                print("out_label_list: %s " % str(out_label_list[i]))
                print("preds_list: %s " % str(preds_list[i]))

if __name__ == '__main__':
    evalute()
