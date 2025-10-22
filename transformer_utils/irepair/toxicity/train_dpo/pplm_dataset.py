"""
Load PPLM dataset
"""
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple

import json
from collections import defaultdict
import random
import os,sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
# from transformer_utils.irepair.toxicity.train_dpo.dpo_utils import get_local_dir, TemporarilySeededRandom
from irepair.constants import DATA_DIR, GPT2_PAD_IDX, DEVICE
# from transformer_utils.src.transformer_utils.location import compute_influence_layer,compute_influence_with_cache
import transformers
import random
import math


def get_pplm_batch_iterator(
    tokenizer,
    config,
    top_data,
    split: str = "train",
    device: str = "cuda",
    batch_size=None,
) -> Iterator[Dict]:
    """
    Get an iterator over batches of data.

    :params:

    :split: Which split to use.
    :batch_size: Batch size.
    :valid_size: Validation size.
    """
    assert split in ["train", "valid"]
    data_dir = os.path.join(DATA_DIR, "toxicity_pairwise")

    if batch_size is None:
        batch_size = config.batch_size
        if split == "valid":
            batch_size = config.eval_batch_size
    max_prompt_length = config.max_prompt_length
    max_new_tokens = config.max_new_tokens
    valid_size = config.valid_size

    filenames = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".jsonl")
    ]

    paired_data = []
    for filename in tqdm(filenames):
        print(filename)
        with open(filename, "r") as file_p:
            file_data = file_p.readlines()

        paired_data.extend(file_data)

    # id="gpt2" #config.model.id
    with open(os.path.join(DATA_DIR, config.model.id+"_utility.json"), 'r') as f:
        util_data = json.load(f)
        util_data = [x["text"] for x in util_data]


    #validation split should remain same across epochs, so shuffule after
    print(len(util_data))
    # print(paired_data)
    if split == "train":
        paired_data = paired_data[:-valid_size]
        util_data = util_data[:-valid_size]
    else:
        paired_data = paired_data[-valid_size:]
        util_data = util_data[-valid_size:]

    random.seed(config.sample_seed)

    random.shuffle(util_data)
    random.shuffle(paired_data)

    # print("xxxxxxxxxxxxxxxxxxxx")
    if split == "train":
        paired_data= top_data
    #     print("xxxxxxxxxxxxxxxxxxxxxxx")
    data_size = len(paired_data)

    for idx in range(0, data_size, batch_size):
        util_text = random.sample(util_data, batch_size)
        util_tokenized = tokenizer(
            util_text,
            max_length=config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        util_input_ids = util_tokenized["input_ids"]
        util_attention_mask = util_tokenized["attention_mask"]

        paired_batch = paired_data[idx : idx + batch_size]
        # print(paired_batch)
        paired_batch = [json.loads(x.strip()) for x in paired_batch if x.strip()]

        prompt_text = [x["prompt_text"] for x in paired_batch]
        gold_text = [x["unpert_gen_text"] for x in paired_batch]
        # gold_text = [x["gold_text"] for x in paired_batch]

        prompt_tokenized = tokenizer(
            prompt_text,
            max_length=max_prompt_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        prompt_input_ids = prompt_tokenized["input_ids"]
        prompt_attention_mask = prompt_tokenized["attention_mask"]

        tokenizer.padding_side = "right"
        gold_tokenized = tokenizer(
            gold_text,
            max_length=max_new_tokens,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        pos_input_id = gold_tokenized["input_ids"].long()

        pplm_text = [x["pert_gen_text"] for x in paired_batch]
        pplm_tokenized = tokenizer(
            pplm_text,
            max_length=max_new_tokens,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        tokenizer.padding_side = "left"

        pos_input_ids = torch.concat(
            [prompt_input_ids, gold_tokenized["input_ids"]], dim=1
        )
        neg_input_ids = torch.concat(
            [prompt_input_ids, pplm_tokenized["input_ids"]], dim=1
        )

        prompt_shape = prompt_input_ids.shape[1]
        pos_labels = pos_input_ids.detach().clone()
        neg_labels = neg_input_ids.detach().clone()

        if hasattr(config.loss, 'skip_prompt_from_loss') and config.loss.skip_prompt_from_loss:
            pos_labels[:, :prompt_shape]=GPT2_PAD_IDX
            neg_labels[:, :prompt_shape] = GPT2_PAD_IDX


        yield {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "gold_text": gold_text,
            "gold_input_ids": pos_input_id,
            "pos_text": gold_text,
            "pos_input_ids": pos_input_ids,
            "pos_attention_mask": pos_input_ids != tokenizer.pad_token_id,
            "pos_labels": pos_labels,
            "neg_text": pplm_text,
            "neg_input_ids": neg_input_ids,
            "neg_attention_mask": neg_input_ids != tokenizer.pad_token_id,
            "neg_labels": neg_labels,
            "util_input_ids": util_input_ids,
            "util_attention_mask": util_attention_mask,
            "util_labels": util_input_ids,
        }
