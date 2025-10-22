"""
Utility functions to save/load models, data, etc.
"""
import json
import os
import random

import torch
from tabulate import tabulate
from datasets import load_dataset,disable_caching
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPTNeoForCausalLM
disable_caching()

def tokenize(tokenizer, data, config):
    """
    Tokenize data.
    """
    max_prompt_size = config.get("max_prompt_size")
    max_new_tokens = config.get("max_new_tokens")
    prompts = [x["prompt"] for x in data]

    print(max_prompt_size)
    if max_prompt_size is not None:
        tokenized = tokenizer(
            prompts,
            max_length=max_prompt_size,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
    elif max_prompt_size is None and len(prompts) == 1:
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
        )
    else:
        raise RuntimeError("Unexpected data tokenization specification.")

    gold = None
    gold_input_ids = None
    gold_attention_mask = None
    if all("gold" in x for x in data):
        gold = [x["gold"] for x in data]
        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        gold_tokenized = tokenizer(
            gold,
            max_length=max_prompt_size + max_new_tokens,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenizer.padding_side = orig_padding_side

        gold_input_ids = gold_tokenized["input_ids"]
        gold_attention_mask = gold_tokenized["attention_mask"]

    return {
        "prompts": prompts,
        "prompt_input_ids": tokenized["input_ids"],
        "prompt_attention_mask": tokenized["attention_mask"],
        "gold": gold,
        "gold_input_ids": gold_input_ids,
        "gold_attention_mask": gold_attention_mask,
    }


def load_model(config):
    """
    Load model, tokenizer.
    """
    assert "model_or_path" in config
    assert "tokenizer" in config

    tokenizer_name = config["tokenizer"]
    model_name = config["model_or_path"]
    state_dict_path = config.get("state_dict_path")
    state_dict = None

        # ["state"]
    # model = HookedTransformer.from_pretrained(model_name)

    if 'gpt2' in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
        state_dict=state_dict
        ).to(config["device"])
    else:
        # model=GPTNeoForCausalLM.from_pretrained(
        model=AutoModelForCausalLM.from_pretrained(
            model_name,
            state_dict=state_dict
        ).to(config["device"])

    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path, map_location=(config["device"]))
        model.load_state_dict(state_dict)

    if tokenizer_name.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id

    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_data(data_config):
    """
    Load data.
    NOTE: Expects a .jsonl file.
    """
    datapath = data_config["datapath"]
    #
    # if datapath.find("edit_data")>-1:
    #     with open(datapath, 'r') as f:
    #         data = json.load(f)
    #     return data

    if datapath.find("edit_data")>-1:

        with open(datapath, "r") as file_p:
            paired_data = json.load(file_p)

        rd=''
        for pd in paired_data:
            rd+=pd["prompt"]+' '+pd["gold"]+'\n\n'
        return [{"prompt": rd}]

    if datapath.endswith(".jsonl"):
        with open(datapath, "r") as file_p:
            data = file_p.readlines()

        data = [json.loads(x.strip()) for x in data]
        return data

    assert "split" in data_config
    print(datapath)
    print(data_config["dataname"])
    print(data_config["split"])
    # data = load_dataset(
    #     datapath, data_config["dataname"], split=data_config["split"]
    # )
    data = load_dataset(
        "parquet", data_dir=datapath,split=data_config["split"]
    )
    # dataset = load_dataset(datapath, split='train')
    # random.seed(88)# Set seed for reproducibility
    # random_indices = random.sample(range(len(dataset)), 2000)
    # data = dataset.select(random_indices)
    return [{"prompt": "\n\n".join(data["text"])}]


def pretty_print_results(results):
    """
    Pretty-print results.
    """
    metrics = None
    reformatted = []
    for intervene_method, _results in results.items():
        if metrics is None:
            metrics = list(_results.keys())

        reformatted.append([intervene_method] + [_results[k] for k in metrics])
        print(_results)

        json_save_path = "/root/autodl-tmp/mnt/project3_dataselection/transformer_utils/irepair/data/result/perplexity.json"
        if os.path.exists(json_save_path):
            with open(json_save_path, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # 追加新结果
        cleaned_results = {}
        for k in metrics:
            v = _results[k]
            if isinstance(v, torch.Tensor):
                cleaned_results[k] = v.item()
            else:
                cleaned_results[k] = v
        print(metrics)
        print(cleaned_results['perplexity'])
        existing_data.append(cleaned_results['perplexity'])

        # 写回 JSON 文件
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

    tabulated = tabulate(reformatted, headers=metrics, tablefmt="orgtbl")
    print(tabulated)




def get_intervene_name(config):
    """
    Construct a name for intervention config.
    """
    name = config["method"]
    if "params" in config:
        params = config["params"]
        if "type" in params:
            name += f"_{params['type']}"
        if "scale" in params:
            name += f"_scale:{params['scale']}"
        if "subtract_from" in params:
            name += f"_subtract_from:{params['subtract_from']}"
    return name
