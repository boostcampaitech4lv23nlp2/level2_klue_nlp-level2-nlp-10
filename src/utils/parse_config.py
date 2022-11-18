import json
from typing import NamedTuple

class ConfigParser(NamedTuple):
    """ Hyperparameters Configuration"""
    seed: int = 777
    project_name: str = "baseline",
    model_name: str = "temp"
    num_labels: int = 30
    
    train_data_path: str = '../dataset/train.csv'
    test_data_path: str = '../dataset/train.csv'
    label_to_num_dict_path: str = 'dict_label_to_num.pkl'
    num_to_label_dict_path: str = 'dict_num_to_label.pkl'
    submission_path: str = '../data/sample_submission.csv'
    
    # directory path
    data_dir: str = "data/"
    save_dir: str = "saved"
    wandb_dir: str = "wandb_checkpoints"
    submission_dir: str = "submissions"
    
    max_epoch: int = 100
    batch_size: int = 128
    shuffle: bool = True 
    validation_split: float = 0.1
    num_workers: int = 2
    lr: float = 5e-5

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))