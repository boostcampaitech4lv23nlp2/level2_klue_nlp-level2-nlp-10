
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModelForSequenceClassification
  

class KLUEModel(pl.LightningModule):
    def __init__(self, conf, device, eval_func=None, is_scheduler=False):
        """
        Args:
            conf (config): configuration file
            device (str): gpu device name
            eval_func (function): metric function
            is_scheduler (bool): scheduler 사용 여부. Defaults to None.
        """
        super().__init__()
        model_config =  AutoConfig.from_pretrained(conf.model_name)
        model_config.num_labels = conf.num_labels
        self._device = device
        self.lr = conf.lr
        self.plm =  AutoModelForSequenceClassification.from_pretrained(conf.model_name, config=model_config)
        self.plm.to(device)
        self.eval_func = eval_func
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.is_scheduler = is_scheduler
        
    def forward(self, x):
        x = self.plm(x['input_ids'], x['attention_mask'], x['token_type_ids'])['logits']
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) 
        loss = self.criterion(logits, y.to(self._device))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.to(self._device))
        self.log("val_loss", loss)
        # if self.eval_func:
        #     self.log("val_metric", self.eval_func(logits.squeeze(), y.squeeze()))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.to(self._device))
        self.log("val_loss", loss)
        # if self.eval_func:
        #     self.log("val_metric", self.eval_func(logits.squeeze(), y.squeeze()))
        return logits

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return logits.squeeze()
        prob = F.softmax(logits, dim= 1)
        result = np.argmax(logits, axis= 1)
        return prob.squeeze(), result.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.is_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=self.lr*0.001)
            return [optimizer], [lr_scheduler]
        else:
            return None
        
        