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


def add_unk_tokens_to_vocab(args, unk_tokens: List[str]) -> None:
    tokenizer_path = args.tokenizer_name.replace("/", "_").replace("-", "_")
    vocab_path = os.path.join(tokenizer_path, "vocab.txt")
    vocab = os.path.join(args.custom_tokenizers_path, vocab_path)
    with open(vocab, "a") as f:
        for token in unk_tokens:
            f.write(token + "\n")


def add_unk_tokens_to_tokenizer(args, unk_tokens: List[str]) -> None:
    tokenizer_path = args.tokenizer_name.replace("/", "_").replace("-", "_")
    tokenizer_config_path = os.path.join(tokenizer_path, "tokenizer.json")
    tokenizer_config = os.path.join(args.custom_tokenizers_path, tokenizer_config_path)
    with open(tokenizer_config, "r") as f:
        data = json.load(f)

    vocab_size = len(data["model"]["vocab"])
    added_token_num = len(unk_tokens)
    new_vocab_index = list(range(vocab_size, vocab_size + added_token_num))
    for idx, tokens in zip(new_vocab_index, unk_tokens):
        data["model"]["vocab"][tokens] = idx

    with open(tokenizer_config, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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
        default="../../tokenizers/",
        help="directory path of custom tokenizers configuration",
    )

    return arg_parser.parse_args()


def main():
    args = get_args()
    df = pd.read_csv(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    indices = get_unk_sentences_idx(df, tokenizer)
    result = get_unk_tokens(df, tokenizer, indices)
    add_unk_tokens_to_vocab(args, result)
    add_unk_tokens_to_tokenizer(args, result)


if __name__ == "__main__":
    main()
