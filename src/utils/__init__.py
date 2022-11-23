import os
import pickle as pickle
import random

import numpy as np
import torch

from .base.dataloader import BaseDataLoader
from .logger import *


def label_to_num(label, dict_path):
    with open(dict_path, "rb") as f:
        dict_label_to_num = pickle.load(f)

    num_label = [dict_label_to_num[v] for v in label]
    return len(dict_label_to_num), num_label


def num_to_label(label, dict_path):
    """숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다."""
    with open(dict_path, "rb") as f:
        dict_num_to_label = pickle.load(f)
    return [dict_num_to_label[v] for v in label]


def set_seed(random_seed):
    print_msg("Setting Seed....", "INFO")
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


def print_config(conf):
    print_msg("Printing Configuration....", "INFO")
    print("=" * 60)
    print("CONFIGURATION")
    print(f"MODEL NAME    | {conf.model_name}")
    print(f"BATCH SIZE    | {conf.batch_size}")
    print(f"MAX EPOCH     | {conf.max_epoch}")
    print(f"SHUFFLE       | {conf.shuffle}")
    print(f"LEARNING RATE | {conf.lr}")
    print(f"SEED          | {conf.seed}")
    print("=" * 60)


def setdir(dirpath, dirname=None, reset=True):
    from shutil import rmtree

    filepath = os.path.join(dirpath, dirname) if dirname else dirpath
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    elif reset:
        print_msg(f"reset directory : {dirname}", "INFO")
        rmtree(filepath)
        os.mkdir(filepath)
    return filepath


def make_file_name(model_name, format, version="v0"):
    file_name = f"{model_name}_{version}.{format}"
    return file_name

