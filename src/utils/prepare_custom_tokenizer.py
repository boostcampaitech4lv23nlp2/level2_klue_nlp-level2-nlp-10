import argparse
import json
import os
from collections import OrderedDict
from typing import List

import numpy as np
import pandas as pd
import transformers
from transformers import AutoModel, AutoTokenizer


def get_unk_sentences_idx(
    df: pd.DataFrame, tokenizer: transformers.AutoTokenizer
) -> List[int]:
    tokenized = list(map(tokenizer.tokenize, df["sentence"]))
    return [
        idx for idx, tokens in enumerate(tokenized) if tokenizer.unk_token in tokens
    ]


def get_unk_tokens(
    df: pd.DataFrame, tokenizer: transformers.AutoTokenizer, indices: List[int]
) -> List[str]:
    unk_token_lst = []

    for idx in indices:
        tokens = tokenizer(df.loc[idx, "sentence"], return_offsets_mapping=True)
        for i, token in enumerate(tokens["input_ids"]):
            if token == tokenizer.unk_token_id:
                start, end = tokens["offset_mapping"][i]
                unk_token_lst.append(df.loc[idx, "sentence"][start:end])
    return list(OrderedDict.fromkeys(unk_token_lst))


def add_unk_tokens_to_vocab(dirpath, unk_tokens: List[str]) -> None:
    vocab_path = os.path.join(dirpath, "vocab.txt")
    with open(vocab_path, "a") as f:
        for token in unk_tokens:
            f.write(token + "\n")


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--tokenizer_name",
        required=True,
        choices=["klue/bert-base", "klue/roberta-large"],
        help="feed tokenizer name",
    )
    arg_parser.add_argument(
        "--data_path",
        required=False,
        default="../../data/train/train.csv",
        help="data which contains unknown tokens",
    )
    arg_parser.add_argument(
        "--custom_tokenizers_path",
        required=False,
        default="../../tokenizers",
        help="directory path of custom tokenizers configuration",
    )

    return arg_parser.parse_args()


def main():
    args = get_args()
    df = pd.read_csv(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    indices = get_unk_sentences_idx(df, tokenizer)
    result = get_unk_tokens(df, tokenizer, indices)
    add_unk_tokens_to_vocab(args.custom_tokenizers_path, result)


if __name__ == "__main__":
    main()
