import os
import pickle as pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from models.metrics import compute_metrics
from models.model import KLUEModel
from pytorch_lightning.loggers import WandbLogger
from utils import *
from utils.dataloader import Dataloader, KLUEDataset


def main(conf, version, is_monitor, is_scheduler):
    print_config(conf)  # configuration parameter 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = setdir(conf.data_dir, conf.save_dir, reset=False)

    # load dataset & dataloader
    train_dataloader = Dataloader(
                            conf.tokenizer_name,
                            conf.train_data_path,
                            conf.label_to_num_dict_path, 
                            max_length = conf.max_length,
                            validation_data_path = conf.validation_data_path,
                            batch_size = conf.batch_size,
                            is_test=False,
                            validation_split=conf.validation_split,
                                )
    # load model
    model = KLUEModel(
        conf, device, eval_func=compute_metrics, is_scheduler=is_scheduler
    )
    model.to(device)

    # learning rate monitoring을 위한 콜백함수 선언
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    model_name = conf.model_name.replace("/", "_")
    if is_monitor == True:
        wandb_logger = WandbLogger(
            project=conf.project_name,
            entity="boost2end",
            save_dir=os.path.join(conf.data_dir, conf.wandb_dir),
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=conf.max_epoch,
            log_every_n_steps=1,
            logger=wandb_logger,
            precision=16,
            callbacks=[lr_monitor],
        )
    else:
        # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=conf.max_epoch,
            log_every_n_steps=1,
            precision=16,
            callbacks=[lr_monitor],
        )

    # Train part
    print_msg("학습을 시작합니다...", "INFO")
    trainer.fit(model=model, datamodule=train_dataloader)
    print_msg("학습이 종료되었습니다...", "INFO")

    # Validation part
    print_msg("최종 모델 검증을 시작합니다...", "INFO")
    trainer.test(model=model, datamodule=train_dataloader)
    print_msg("최종 모델 검증이 종료되었습니다...", "INFO")

    # 학습이 완료된 모델의 state dict 저장
    print_msg("마지막 모델을 저장합니다...", "INFO")
    file_name = make_file_name(model_name, format="pt", version=version)
    model_path = os.path.join(save_path, file_name)
    torch.save(model.state_dict(), model_path)

