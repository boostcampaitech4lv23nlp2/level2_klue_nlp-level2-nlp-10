import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils import *
import pytorch_lightning as pl


os.environ["TOKENIZERS_PARALLELISM"] = "false"

class KLUEDataset(Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, dataset, labels):
    self.dataset = dataset 
    self.labels = labels
    
  def __len__(self):
    if self.labels:
      return len(self.labels)
    else:
      return len(self.dataset)
  
  def __getitem__(self, idx):
    data = {key: val[idx].clone().detach() for key, val in self.dataset.items()}
    if self.labels:
      labels = torch.tensor(self.labels[idx])
      return (data, labels)
    else:
      return data


class Dataloader(pl.LightningDataModule):
  """ Dataset을 불러오기 위한 dataloader class."""

  def __init__(self, model_name, data_path, label_dict_path, batch_size=64, is_test=False):
    """_summary_

    Args:
        model_name (str): pretrained 모델명 ex) klue/bert-base
        data_path (str): 학습 및 테스트 데이터 경로 
        label_dict_path (str): label_to_num 함수에서 호출되는 dictionary 경로 
        batch_size (int, optional): batch size
        is_test (bool, optional): 학습을 위한 dataloader인지 추론을 위한 dataloader인지 확인하기 위한 인자
    """
    super().__init__()
    self.is_test = is_test
    self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=160)
    self.batch_size = batch_size
    self.label_dict_path = label_dict_path
    self.data_path = data_path
    self.dataset = self.setup()
    if not self.is_test:
      self.dataloader = BaseDataLoader(self.dataset, 
                                        batch_size=self.batch_size,              
                                        shuffle=True,
                                        num_workers = 4, 
                                        is_test=self.is_test, 
                                        validation_split=0.2)
    else:
      self.dataloader = DataLoader(self.dataset, 
                                   batch_size=self.batch_size,
                                   num_workers = 4, )
  def train_dataloader(self):
    return self.dataloader
  
  def val_dataloader(self):
    if self.is_test:
      return self.dataloader 
    else:
      return self.dataloader.val_dataloader()
  
  def test_dataloader(self):
    if self.is_test:
      return self.dataloader 
    else:
      return self.dataloader.val_dataloader()
  
  def predict_dataloader(self):
    return self.dataloader
  
  def load_data(self, data_path):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(data_path)
    return pd_dataset
  
  def setup(self, stage='fit'):
    print_msg('Loading Dataset...', 'INFO')
    dataset = self.load_data(self.data_path)
    dataset, labels = self.preprocessing(dataset)
    dataset = KLUEDataset(dataset, labels)
    return dataset
  
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
    tokenized_dataset = self.tokenize_dataset(out_dataset)
    if not self.is_test:
      str_label = dataset['label'].values
      self.num_labels, num_label = label_to_num(str_label, self.label_dict_path)
      return tokenized_dataset, num_label
    return tokenized_dataset, None

  
  


