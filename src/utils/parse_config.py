import json
from typing import NamedTuple

class ConfigParser(NamedTuple):
    """ Hyperparameters Configuration"""
    seed: int = 777
    model_name: str = "temp"
    n_gpu: int = 1
    
    data_path: str = '../dataset/train.csv'
    label_dict_path: str = 'dict_label_to_num.pkl'
    
    # directory path
    data_dir: str = "data/"
    save_dir: str = "saved"
    
    batch_size: int = 128
    shuffle: bool = True 
    validation_split: float = 0.1
    num_workers: int = 2
    lr: float = 1e-4
    num_labels: int = 30
    
    optimizer_name: str = "AdamW"
    loss: str = "nll_loss"
    metrics: list = ["accuracy", "top_k_acc"]
    
    max_epoch: int = 100
    
    save_period: int = 1
    
    is_scheduler: bool = False
    is_early_stopping: bool = False
    is_monitor: bool = True
    

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))