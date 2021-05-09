import os
import time
import torch
from cfg import *

def compute_f_score(tp, fn, fp):
    r"""
    :param tp: int, true positive
    :param fn: int, false negative
    :param fp: int, false positive
    :return: (f, pre, rec)
    """
    epsilon = 1e-13
    pre = tp / (fp + tp + epsilon)
    rec = tp / (fn + tp + epsilon)
    f = 2 * pre * rec / (pre + rec + 1e-13)

    return f, pre, rec

def save_model(model, path_prefix=None, save_path=None, score=None):
    if path_prefix is not None:
        save_dir = os.path.join(MODEL_PATH, path_prefix)
    else:
        save_dir = MODEL_PATH
    if save_path is None:
        if score is None:
            save_path = os.path.join(save_dir, f'{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))}.pt')
        else:
            save_path = os.path.join(save_dir, f'{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))}_{score}.pt')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print('save checkpoint {} .'.format(save_path))
    torch.save(model.state_dict(), save_path)
    return save_path