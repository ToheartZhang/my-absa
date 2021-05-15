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
from ate.evaluate import ate_evaluate
from asc.evaluate import asc_evaluate
from cfg import *
from utils import compute_f_score, save_model

if __name__ == '__main__':
    matched_ids = ate_evaluate()
    print(f'Total matched term: {len(matched_ids)}')
    asc_evaluate(matched_ids)
