import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np

from utils.dataloader import KLUEDataset, KFoldDataloader, Dataloader
from utils import *
from models.metrics import compute_metrics
from models.model import KLUEModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.metrics import compute_metrics

def main(conf, version, is_monitor, is_scheduler):
    print_config(conf) # configuration parameter 확인 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = setdir(conf.data_dir, conf.save_dir, reset=False)
    
    # learning rate monitoring을 위한 콜백함수 선언
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    model_name = conf.model_name.replace('/', '_')
    
    if is_monitor==True:
        wandb_logger = WandbLogger(project = conf.project_name, entity='boost2end',
                                   save_dir = os.path.join(conf.data_dir, conf.wandb_dir))
        trainer = pl.Trainer(accelerator='gpu', devices=1,
                             max_epochs=conf.max_epoch, log_every_n_steps=1, logger=wandb_logger,
                             precision=16,
                             callbacks=[lr_monitor])
    else:
        # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
        trainer = pl.Trainer(accelerator='gpu', devices=1,
                            max_epochs=conf.max_epoch, log_every_n_steps=1,
                            precision=16,
                            callbacks=[lr_monitor])

    # Train part
    results = []
    for k in range(conf.num_folds):
        # load dataset & dataloader    
        train_dataloader = KFoldDataloader(
                                            conf.tokenizer_name,
                                            conf.train_data_path,
                                            conf.label_to_num_dict_path, 
                                            seed = conf.seed,
                                            max_length = conf.max_length,
                                            batch_size = conf.batch_size,
                                            is_test = False,
                                            k = k,
                                            num_folds = conf.num_folds
                                            )
        model = KLUEModel(conf, device, eval_func = compute_metrics, 
                                    is_scheduler = is_scheduler)
        model.to(device)

        print_msg(f'fold {k} 학습을 시작합니다...', 'INFO')
        trainer.fit(model=model, datamodule=train_dataloader)
        print_msg(f'fold {k} 학습이 종료되었습니다...', 'INFO')
    
        # Validation part
        print_msg(f'fold {k} 모델 검증을 시작합니다...', 'INFO')
        score = trainer.test(model=model, datamodule=train_dataloader)
        print_msg(f'fold {k} 모델 검증이 종료되었습니다...', 'INFO')
        results.extend(score)
    
    
    final_micro_f1 = [ret['micro f1 score'] for ret in results]
    final_auprc = [ret['auprc'] for ret in results]
    final_accuracy = [ret['accuracy'] for ret in results]
    if is_monitor == True:
        model.log({"final_micro_f1": final_micro_f1})
        model.log({"final_auprc": final_auprc})
        model.log({"final_accuracy": final_accuracy})
    print_msg(f"K fold 검증 결과 : (f1){final_micro_f1}, \
                                 (auprc){final_auprc}, \
                                 (acc){final_accuracy}", "INFO")
    
    
    # 전체 데이터를 사용하여 학습을 진행합니다. 
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
    model = KLUEModel(conf, 
                      device, 
                      eval_func = compute_metrics,
                      is_scheduler = is_scheduler
                      )
    model.to(device)

    # Train part
    print_msg('학습을 시작합니다...', 'INFO')
    trainer.fit(model=model, datamodule=train_dataloader)
    print_msg('학습이 종료되었습니다...', 'INFO')
    
    # 학습이 완료된 모델의 state dict 저장
    print_msg('마지막 모델을 저장합니다...', 'INFO')
    file_name = make_file_name(model_name, format='pt', version=version)
    model_path = os.path.join(save_path, file_name)
    torch.save(model.state_dict(), model_path)
  

  