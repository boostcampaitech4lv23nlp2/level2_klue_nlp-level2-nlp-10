import os
import random
import torch
import numpy as np
from .parse_config import *
from .logger import *
from .base_dataloader import BaseDataLoader

def set_seed(random_seed):
    print_msg('Setting Seed....', 'INFO')
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    
def print_config(conf):
    print('='*60)
    print_msg('Print Configuration....', 'INFO')
    print(f'MODEL NAME    | {conf.model_name}')
    print(f'BATCH SIZE    | {conf.batch_size}')
    print(f'MAX EPOCH     | {conf.max_epoch}')
    print(f'SHUFFLE       | {conf.shuffle}')
    print(f'LEARNING RATE | {conf.lr}')
    print(f'SEED          | {conf.seed}')
    print('='*60)
    
def setdir(dirpath, dirname=None, reset=True):
    from shutil import rmtree
    filepath = os.path.join(dirpath, dirname) if dirname else dirpath      
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    elif reset:
        print(f"reset directory : {dirname}")
        rmtree(filepath)
        os.mkdir(filepath)
    return filepath 