import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np

from .dataloader import KLUEDataset, Dataloader
from utils import *
from models.metrics import compute_metrics
from models.model import KLUEModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

def main(conf, is_monitor):
    print_config(conf)
    set_seed(conf.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    saved_path = setdir(conf.data_dir, conf.save_dir, reset=False)

    # load dataset & dataloader    
    
    train_dataloader = Dataloader(conf.model_name, 
                                  conf.data_path,
                                  conf.label_dict_path, 
                                  conf.batch_size)
    model = KLUEModel(conf, device, train_dataloader.num_labels)
    
    # learning rate monitoring
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    if is_monitor:
        wandb_logger = WandbLogger(project=conf.model_name.replace('/', '_'))
        trainer = pl.Trainer(accelerator='gpu', devices=1,
                             max_epochs=conf.max_epoch, log_every_n_steps=1, logger=wandb_logger,
                            #  precision=16,
                             callbacks=[lr_monitor])
    else:
        # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
        trainer = pl.Trainer(accelerator='gpu', devices=1,
                            max_epochs=conf.max_epoch, log_every_n_steps=1,
                            # precision=16,
                            callbacks=[lr_monitor])

    # Train part
    trainer.fit(model=model, datamodule=train_dataloader)
    trainer.test(model=model, datamodule=train_dataloader)
  

  