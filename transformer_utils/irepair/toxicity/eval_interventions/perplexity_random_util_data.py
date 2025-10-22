import json
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import float16
from transformers import AutoModelForCausalLM, GPT2Tokenizer

from constants import DATA_DIR, DEVICE, ROOT_DIR, GPT2_PAD_IDX
from util.torch_layer_util import get_batch_logps

torch.backends.cuda.matmul.allow_tf32 = True


def argmax_gen(model, tokenizer, batch):
    # Start with the initial input text
    l=batch['util_input_ids'].shape[1]
    prompt_length=int(l/2)
    prompt=batch['util_input_ids'][:, :prompt_length]

    max_length = l-prompt_length

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids=prompt).logits.to(torch.float32)
            logits = outputs[:, -1, :]

            # Select the token with the highest probability (argmax)
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)

            # Append the selected token to the input_ids
            prompt = torch.cat([prompt, next_token_id], dim=-1)

            # Stop if the end of sequence token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    # Decode the generated input_ids to text
    prompt=prompt.squeeze()
    prompt_text = tokenizer.decode(prompt[0:prompt_length], skip_special_tokens=True)
    continuation_text = tokenizer.decode(prompt[prompt_length:], skip_special_tokens=True)

    return {"prompt_text": prompt_text, "continuation_text": continuation_text}


def get_util_test_raw(
        tokenizer,
        batch_size=8,
        toxicity_cutoff=0.2
):
    with open(os.path.join(DATA_DIR, "utility", "test.jsonl"), 'r') as f:
        util_data = json.load(f)

    util_data = [x for x in util_data if x['score'] < toxicity_cutoff]

    data_size = len(util_data)

    for idx in range(0, data_size, batch_size):
        util_batch = util_data[idx: idx + batch_size]
        util_text = [x["text"] for x in util_batch]

        util_tokenized = tokenizer(
            util_text,
            max_length=101,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        util_input_ids = util_tokenized["input_ids"]
        util_attention_mask = util_tokenized["attention_mask"]

        yield {
            "util_input_ids": util_input_ids,
            "util_attention_mask": util_attention_mask,
        }

def get_util_test_prompts(
        tokenizer,
        batch_size=8,
):
    with open(os.path.join(DATA_DIR, "utility", "test_prompt.jsonl"), 'r') as f:
        util_data = json.load(f)

    data_size = len(util_data)

    for idx in range(0, data_size, batch_size):
        util_batch = util_data[idx: idx + batch_size]
        util_prompts = [x["prompt_text"] for x in util_batch]
        util_conts = [x["continuation_text"] for x in util_batch]

        p_tokenized = tokenizer(
            util_prompts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)
        tokenizer.padding_side = "right"
        c_tokenized = tokenizer(
            util_conts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)
        tokenizer.padding_side = "left"

        concatenated_input_ids = []
        concatenated_attention_mask = []
        concatenated_labels=[]

        # Concatenate input_ids and attention_mask for each pair
        for p_input_ids, c_input_ids, p_attention_mask, c_attention_mask in zip(
                p_tokenized["input_ids"], c_tokenized["input_ids"],
                p_tokenized["attention_mask"], c_tokenized["attention_mask"]):
            concatenated_input_ids.append(torch.cat([p_input_ids, c_input_ids], dim=-1))
            concatenated_attention_mask.append(torch.cat([p_attention_mask, c_attention_mask], dim=-1))
            label=torch.full_like(p_input_ids, GPT2_PAD_IDX)
            concatenated_labels.append(torch.cat([label, c_input_ids], dim=-1))

        # Stack the concatenated sequences into tensors
        concatenated_input_ids = torch.nn.utils.rnn.pad_sequence(concatenated_input_ids, batch_first=True,
                                                                 padding_value=GPT2_PAD_IDX)
        concatenated_labels = torch.nn.utils.rnn.pad_sequence(concatenated_labels, batch_first=True,
                                                                 padding_value=GPT2_PAD_IDX)
        concatenated_attention_mask = concatenated_input_ids!=GPT2_PAD_IDX
        yield {
            "util_input_ids": concatenated_input_ids.to(DEVICE),
            "util_attention_mask": concatenated_attention_mask.to(DEVICE),
            "util_labels": concatenated_labels.to(DEVICE)
        }

def compute_perpexility(model_name, state_dict_path=None):
    state_dict = None
    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path, map_location=(DEVICE))["state"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        state_dict=state_dict,
    ).to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    test_iterator = get_util_test_prompts(tokenizer)

    ppl = 0.0
    nb = 0
    lls=[]
    for batch in test_iterator:
        with torch.no_grad():
            logits = model(
                batch["util_input_ids"],
                attention_mask=batch["util_attention_mask"],
            ).logits.to(torch.float32)
            loss = -1.0 * get_batch_logps(logits, batch["util_labels"], token_wise_mean=True)
            ppl += loss
            nb += 1

    return ppl/nb

def continue_prompt(model_name, state_dict_path=None):
    state_dict = None
    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path, map_location=(DEVICE))["state"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        state_dict=state_dict,
    ).to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    test_iterator = get_util_test_raw(tokenizer, batch_size=1)
    nt=[]
    for batch in test_iterator:
        nt.append(argmax_gen(model, tokenizer, batch))
    with open(os.path.join(ROOT_DIR, "data", "utility", "test_prompt.jsonl"), 'w', encoding='utf-8') as f:
        json.dump(nt, f)


if __name__ == "__main__":
    model_name='gpt2-medium'
    # state_dict_path= os.path.join(ROOT_DIR, 'checkpoints', "policy.pt")
    state_dict_path=None
    print('Perplexity: ', compute_perpexility(model_name, state_dict_path))
    # continue_prompt(model_name, state_dict_path)