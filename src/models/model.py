import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
)


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
        self.save_hyperparameters()
        model_config = AutoConfig.from_pretrained(conf.model_name)
        model_config.num_labels = conf.num_labels
        self._device = device
        self.lr = conf.lr
        self.max_length = conf.max_length
        self.plm = AutoModelForSequenceClassification.from_pretrained(
            conf.model_name, config=model_config, local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            conf.tokenizer_name, max_length=self.max_length
        )
        self.plm.resize_token_embeddings(self.tokenizer.vocab_size)
        # self.plm.resize_token_embeddings(model_config.vocab_size + 16)
        self.eval_func = eval_func
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.is_scheduler = is_scheduler

    def forward(self, x):
        x = self.plm(x["input_ids"], x["attention_mask"], x["token_type_ids"])["logits"]
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

        if self.eval_func:
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, 1)
            ret = self.eval_func(
                probs.squeeze().cpu().detach().numpy(),
                preds.squeeze().cpu().detach().numpy(),
                y.squeeze().cpu().detach().numpy(),
            )
            self.log("micro_f1_score", ret["micro_f1_score"])
            self.log("auprc", ret["auprc"])
            self.log("accuracy", ret["accuracy"])
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.to(self._device))
        self.log("val_loss", loss)

        if self.eval_func:
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, 1)
            ret = self.eval_func(
                probs.squeeze().cpu().detach().numpy(),
                preds.squeeze().cpu().detach().numpy(),
                y.squeeze().cpu().detach().numpy(),
            )
            self.log("micro_f1_score", ret["micro_f1_score"])
            self.log("auprc", ret["auprc"])
            self.log("accuracy", ret["accuracy"])
        return logits

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        return probs.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.is_scheduler:
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=self.lr*0.001)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=1, T_mult=2, eta_min=self.lr * 0.01
            )
            return [optimizer], [lr_scheduler]
        else:
            return optimizer
