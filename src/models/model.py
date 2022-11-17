
import torch
import torch.nn as nn 
import numpy as np 
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModelForSequenceClassification
  

class KLUEModel(pl.LightningModule):
    def __init__(self, conf, device, num_labels, criterion=None, eval_func=None, scheduler=None):
        super().__init__()
        model_config =  AutoConfig.from_pretrained(conf.model_name)
        model_config.num_labels = num_labels
        self._device = device
        self.lr = conf.lr
        self.plm =  AutoModelForSequenceClassification.from_pretrained(conf.model_name, config=model_config)
        self.plm.to(device)
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.plm(x)['logits']
        # softmax(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x = batch['input_ids']
        mask = batch['attention_mask']
        y = batch['labels']
        
        logits = self(x)
        loss = self.criterion(logits, y.to(self._device))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['input_ids']
        mask = batch['attention_mask']
        y = batch['labels']
        logits = self(x)
        loss = self.criterion(logits, y.to(self._device))
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)


    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.scheduler:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=self.lr*0.001)
            return [optimizer], [lr_scheduler]
        else:
            return None
        
        