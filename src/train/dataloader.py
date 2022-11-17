from abc import abstractmethod
from numpy import inf

import pandas as pd
import pickle as pickle
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils import *
import pytorch_lightning as pl

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class KLUEDataset(Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, dataset, labels):
    self.dataset = dataset 
    self.labels = labels
    
  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item


class Dataloader(pl.LightningDataModule):
  def __init__(self, model_name, data_path, label_dict_path, batch_size=64, is_test=False):
    super().__init__()
    self.tokenizer = AutoTokenizer.from_pretrained(model_name, )
    self.batch_size = batch_size
    self.label_dict_path = label_dict_path
    self.data_path = data_path
    self.dataset = self.setup()
    self.dataloader = BaseDataLoader(self.dataset, 
                                      batch_size=self.batch_size,              
                                      shuffle=True,
                                      num_workers = 4, 
                                      is_test=is_test, 
                                      validation_split=0.2)
  def train_dataloader(self):
    return self.dataloader
  
  def val_dataloader(self):
    return self.dataloader.val_dataloader()
  
  def test_dataloader(self):
    return self.dataloader
  
  def predict_dataloader(self):
    return self.dataloader
    
  def preprocessing(self, dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
      i = i[1:-1].split(',')[0].split(':')[1]
      j = j[1:-1].split(',')[0].split(':')[1]

      subject_entity.append(i)
      object_entity.append(j)
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
    return out_dataset

  def load_data(self, data_path):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(data_path)
    dataset = self.preprocessing(pd_dataset)
    return dataset
  
  def label_to_num(self, label):
    num_label = []
    with open(self.label_dict_path, 'rb') as f:
      dict_label_to_num = pickle.load(f)
      
    for v in label:
      num_label.append(dict_label_to_num[v])
    self.num_labels = len(dict_label_to_num)
    return num_label
  
  def tokenize_dataset(self, dataset):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
      temp = ''
      temp = e01 + '[SEP]' + e02
      concat_entity.append(temp)

    tokenized_sentences = self.tokenizer(concat_entity,
                                          list(dataset['sentence']),
                                          return_tensors="pt",
                                          padding=True,
                                          truncation=True,
                                          max_length=256,
                                          add_special_tokens=True,
                                          )
    return tokenized_sentences
  
  def setup(self, stage='fit'):
    print_msg('Loading Dataset...', 'INFO')
    dataset = self.load_data(self.data_path)
    str_label = dataset['label'].values
    num_label = self.label_to_num(str_label)
    tokenized_dataset = self.tokenize_dataset(dataset)
    dataset = KLUEDataset(tokenized_dataset, num_label)
    return dataset


