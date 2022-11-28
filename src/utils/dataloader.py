import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import KFold, StratifiedKFold
from utils import *
import pytorch_lightning as pl


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class KLUEDataset(Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        self.kfold_index = None

    def __len__(self):
        if self.kfold_index:
            return len(self.kfold_index)
        elif self.labels:
            return len(self.labels)
        else:
            return len(self.dataset["input_ids"])

    def set_kfold_index(self, indexes):
        self.kfold_index = indexes

    def get_kfold_index(self):
        return self.kfold_index

    def __getitem__(self, idx):
        if self.kfold_index:
            item_idx = self.kfold_index[idx]
        else:
            item_idx = idx

        data = {
            key: val[item_idx].clone().detach() for key, val in self.dataset.items()
        }
        if self.labels:
            labels = torch.tensor(self.labels[item_idx])
            return (data, labels)
        else:
            return data


class Dataloader(pl.LightningDataModule):
    """Dataset을 불러오기 위한 dataloader class."""

    def __init__(
        self,
        tokenizer_name,
        data_path,
        label_dict_path,
        max_length=256,
        validation_data_path=None,
        batch_size=64,
        is_test=False,
        validation_split=0.1,
    ):
        """
        Args:
            model_name (str): pretrained 모델명 ex) klue/bert-base
            data_path (str): 학습 및 테스트 데이터 경로
            label_dict_path (str): label_to_num 함수에서 호출되는 dictionary 경로
            validation_data_path (str) : validation dataset이 존재할 경우의 파일 경로
            batch_size (int, optional): batch size
            is_test (bool, optional): 학습을 위한 dataloader인지 추론을 위한 dataloader인지 확인하기 위한 인자
        """
        super().__init__()
        self.is_test = is_test
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, max_length=self.max_length
        )
        self.batch_size = batch_size
        self.label_dict_path = label_dict_path

        self.data_path = data_path
        self.validation_data_path = validation_data_path
        self.validation_split = validation_split

        self.dataset = None
        self.val_dataset = None
        self.dataloader = None

    def train_dataloader(self):
        return (
            self.dataloader
            if self.dataloader
            else DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=4,
                collate_fn=default_collate,
            )
        )

    def val_dataloader(self):
        return (
            self.dataloader.val_dataloader()
            if self.dataloader
            else DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=4,
                collate_fn=default_collate,
            )
        )

    def test_dataloader(self):
        return (
            self.dataloader.val_dataloader()
            if self.dataloader
            else DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=4,
                collate_fn=default_collate,
            )
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=default_collate,
        )

    def load_data(self, data_path):
        """csv 파일을 경로에 맡게 불러 옵니다."""
        pd_dataset = pd.read_csv(data_path)
        return pd_dataset

    def setup(self, stage="fit"):

        print_msg("Loading Dataset...", "INFO")
        dataset = self.load_data(self.data_path)
        dataset, labels = self.preprocessing(dataset)
        self.dataset = KLUEDataset(dataset, labels)

        if stage == "fit":
            print_msg("Loading validation Dataset...", "INFO")
            val_dataset = None
            if self.validation_data_path:
                val_dataset = self.load_data(self.validation_data_path)
                val_dataset, val_labels = self.preprocessing(val_dataset)
                self.val_dataset = KLUEDataset(val_dataset, val_labels)

            if not self.val_dataset:  # random split validation dataset
                print_msg("무작위로 validation dataset을 선별합니다...", "INFO")
                self.dataloader = BaseDataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=4,
                    is_test=self.is_test,
                    validation_split=self.validation_split,
                )

    def tokenize_dataset(self, dataset):
        """tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = []
        for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
            temp = ""
            temp = e01 + "[SEP]" + e02
            concat_entity.append(temp)

        tokenized_sentences = self.tokenizer(
            concat_entity,
            list(dataset["sentence"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        return tokenized_sentences

    def preprocessing(self, dataset):
        """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""

        subject_entity = []
        object_entity = []
        for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
            i = i[1:-1].split(",")[0].split(":")[1]
            j = j[1:-1].split(",")[0].split(":")[1]
            subject_entity.append(i)
            object_entity.append(j)

        out_dataset = pd.DataFrame(
            {
                "id": dataset["id"],
                "sentence": dataset["sentence"],
                "subject_entity": subject_entity,
                "object_entity": object_entity,
                "label": dataset["label"],
            }
        )
        tokenized_dataset = self.tokenize_dataset(out_dataset)

        if not self.is_test:
            str_label = dataset["label"].values
            _, num_label = label_to_num(str_label, self.label_dict_path)
            return tokenized_dataset, num_label
        return tokenized_dataset, None


class KFoldDataloader(Dataloader):
    def __init__(
        self,
        tokenizer_name,
        data_path,
        label_dict_path,
        seed=3431,
        max_length=256,
        validation_data_path=None,
        batch_size=64,
        is_test=False,
        validation_split=0.1,
        k=0,
        num_folds=5,
    ):

        super().__init__(
            tokenizer_name,
            data_path,
            label_dict_path,
            max_length=max_length,
            validation_data_path=validation_data_path,
            batch_size=batch_size,
            is_test=is_test,
            validation_split=validation_split,
        )

        self.k = k
        self.num_folds = num_folds
        self.seed = seed

    def setup(self, stage="fit"):
        dataset, val_dataset = None, None
        print_msg("Loading Dataset...", "INFO")
        dataset = self.load_data(self.data_path)
        dataset, labels = self.preprocessing(dataset)
        self.dataset = KLUEDataset(dataset, labels)
        self.val_dataset = KLUEDataset(dataset, labels)

        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        all_splits = [k for k in kf.split(self.dataset)]

        # fold한 index에 따라 데이터셋 분할
        train_indexes, val_indexes = all_splits[self.k]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

        self.dataset.set_kfold_index(train_indexes)
        self.val_dataset.set_kfold_index(val_indexes)
