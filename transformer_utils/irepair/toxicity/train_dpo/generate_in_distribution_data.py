import json
import os
import random

import hydra
import torch
import transformers
from omegaconf import OmegaConf, DictConfig
from transformers import GPTNeoForCausalLM

# from constants import DEVICE, ROOT_DIR
# from toxicity.train_dpo.dpo_utils import get_local_dir, get_local_run_dir, disable_dropout
import os,sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from irepair.constants import DEVICE, ROOT_DIR
from irepair.toxicity.train_dpo.dpo_utils import get_local_dir, get_local_run_dir, disable_dropout
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
OmegaConf.register_new_resolver(
    "get_local_run_dir",
    lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs),
)

def nucleus_sampling(logits, top_p=0.9, temperature=1.0):
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[..., indices_to_remove] = -float('Inf')
    softmax_probs = torch.softmax(logits, dim=-1)
    next_token_id = torch.multinomial(softmax_probs, num_samples=1)

    return next_token_id


def custom_gen(model, tokenizer, seed=None):
    # Generate a random seed for each call
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Start with the SOS token
    input_ids = torch.tensor([[tokenizer.eos_token_id]], device=model.device)
    # input_ids = tokenizer.encode("i like to", return_tensors='pt').to(model.device)

    max_length = 101

    for _ in range(max_length):
        outputs = model(input_ids=input_ids).logits.to(torch.float32)
        logits = outputs[:, -1, :]

        next_token_id = nucleus_sampling(logits, top_p=0.9, temperature=1.0)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    # print(f"Generated with seed {seed}: {generated_text}")
    return {"text": generated_text}
def save_json_split(data, split_size_mb=118):
    split_size_bytes = split_size_mb * 1024 * 1024  # Convert MB to bytes
    split_counter = 0
    current_split = []
    current_size = 0

    for item in data:
        item_str = json.dumps(item) + '\n'
        item_size = len(item_str.encode('utf-8'))

        if current_size + item_size > split_size_bytes:
            with open(os.path.join(ROOT_DIR,"data", "utility", str(split_counter)+".json"), 'w', encoding='utf-8') as f:
                json.dump(current_split, f)
            split_counter += 1
            current_split = []
            current_size = 0

        current_split.append(item)
        current_size += item_size

    if current_split:  # Save the remaining data
        with open(os.path.join(ROOT_DIR, "data", "utility", str(split_counter) + ".json"), 'w', encoding='utf-8') as f:
            json.dump(current_split, f)



@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):

    OmegaConf.resolve(config)
    os.environ["XDG_CACHE_HOME"] = get_local_dir(config.local_dirs)
    model_kwargs = (
        {"device_map": "balanced"} if config.trainer == "BasicTrainer" else {}
    )
    policy_dtype = getattr(torch, config.model.policy_dtype)
    if 'gpt2' or 'qwen' or "pythia" in config.model.name_or_path:
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            cache_dir=get_local_dir(config.local_dirs),
            low_cpu_mem_usage=True,
            torch_dtype=policy_dtype,
            **model_kwargs,
        )
    else:
        policy=GPTNeoForCausalLM.from_pretrained(
            config.model.name_or_path,
            cache_dir=get_local_dir(config.local_dirs),
            low_cpu_mem_usage=True,
            torch_dtype=policy_dtype,
            **model_kwargs,
        )
    disable_dropout(policy)
    policy=policy.to(DEVICE)

    tokenizer_name_or_path = (
            config.model.tokenizer_name_or_path or config.model.name_or_path
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs)
        )

    # Parameters
    num_texts = 30000
    split_size_mb = 200
    #
    # current_split = []
    # for i in range(num_texts):
    #     # Generate text
    #     generated_text = custom_gen(policy, tokenizer)
    #
    #     # Convert to JSON format
    #     item_str = json.dumps(generated_text) + '\n'
    #
    #     # Add the new text to the current split
    #     current_split.append(generated_text)
    #
    #     # Print progress
    #     if (i + 1) % 1000 == 0:
    #         print(f"Generated {i + 1}/{num_texts} texts")
    #
    #     split_file_path = os.path.join(ROOT_DIR, "data", "utility", f"gpt2_large_train.jsonl")
    #     with open(split_file_path, 'w', encoding='utf-8') as f:
    #         json.dump(current_split, f)
    split_file_path = os.path.join(ROOT_DIR, "data", "pythia_utility.json")
    print(ROOT_DIR)

    # ==== 加载已有部分（断点续写） ====
    if os.path.exists(split_file_path):
        with open(split_file_path, 'r', encoding='utf-8') as f:
            current_split = json.load(f)
        start_idx = len(current_split)
        print(f"Resuming from {start_idx}/{num_texts} texts")
    else:
        current_split = []
        start_idx = 0

    # ==== 开始生成 ====
    for i in range(start_idx, num_texts):
        generated_text = custom_gen(policy, tokenizer)
        current_split.append(generated_text)

        if (i + 1) % 1000 == 0 or (i + 1) == num_texts:
            print(f"Generated {i + 1}/{num_texts} texts, saving to file...")
            os.makedirs(os.path.dirname(split_file_path), exist_ok=True)
            with open(split_file_path, 'w', encoding='utf-8') as f:
                json.dump(current_split, f)


if __name__ == "__main__":
    main()
