import os
import re
import random
import numpy as np
import hydra
import torch
import transformers
from omegaconf import OmegaConf, DictConfig
from transformers import AutoModelForSeq2SeqLM, GPTNeoForCausalLM
import os,sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

from irepair.constants import DEVICE
from dpo_utils import get_local_run_dir, get_local_dir, disable_dropout
from pplm_dataset import get_pplm_batch_iterator
from irepair.util.aws_util import save_sensitivity, load_sensitivity
from irepair.util.torch_layer_util import get_trainable_layers, get_batch_logps


def compute_sensitivity(model, data, num_sample, type='neg'):
    example_counter = 0
    gradients = {}
    for example in data:
        if example_counter >= num_sample:
            break
        example_counter += 1
        logits = model(
            example[type + "_input_ids"],
            attention_mask=example[type + "_attention_mask"],
        ).logits.to(torch.float32)
        loss = -1.0 * get_batch_logps(logits, example[type + "_labels"], token_wise_mean=True)
        loss.backward()

        for name, param in model.named_parameters():
            if name not in gradients:
                gradients[name] = 0.0
            gradients[name] += param.grad

        model.zero_grad()

    for name, param in model.named_parameters():
        gradients[name] /= num_sample
    sensitivities = {}
    for name in gradients.keys():
        match = re.match(r'^(.*?\.\d+)', name)
        layer_name = match.group(0) if match else None
        if layer_name is None:
            continue
        if layer_name not in sensitivities:
            sensitivities[layer_name] = []
        sensitivities[layer_name].append(gradients[name].norm(2).item() ** 2)

    for layer in sensitivities:
        sensitivities[layer] = sum(sensitivities[layer]) ** 0.5

    sensitivities = dict(sorted(sensitivities.items(), key=lambda item: item[1]))

    for layer in sensitivities:
        print(layer + ',' + str(sensitivities[layer]));

    mtl = max(sensitivities, key=sensitivities.get)
    print('Max layer name for ' + type + ': ' + mtl)
    return sensitivities


def init_fixed_selection(config, model):
    # mtl = 'transformer.h.0'  # for gpt2-xl
    mtl='transformer.h.3' #for gpt neo
    # mtl = 'transformer.h.5'  # gpt2-large
    for paramName, param in model.named_parameters():
        match = re.match(r'^(.*?\.\d+)', paramName)
        layer_name = match.group(0) if match else None
        if layer_name != mtl:
            param.requires_grad = False
        else:
            param.requires_grad = True


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    os.environ["XDG_CACHE_HOME"] = get_local_dir(config.local_dirs)
    model_kwargs = (
        {"device_map": "balanced"} if config.trainer == "BasicTrainer" else {}
    )
    policy_dtype = getattr(torch, config.model.policy_dtype)

    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        cache_dir=get_local_dir(config.local_dirs),
        low_cpu_mem_usage=True,
        torch_dtype=policy_dtype,
        **model_kwargs,
    )

    disable_dropout(policy)
    policy = policy.to(DEVICE)

    np.random.seed(87)
    random.seed(87)

    tokenizer_name_or_path = (
            config.model.tokenizer_name_or_path or config.model.name_or_path
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs)
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_sample = config.loss.num_sample_fixed_selection
    train_iterator = get_pplm_batch_iterator(
        tokenizer, config, split="train", device=DEVICE)
    sen1 = compute_sensitivity(policy, train_iterator, num_sample, type="neg")
    save_sensitivity(sen1, config.model.name_or_path + "_neg_raw_" + str(num_sample))


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    OmegaConf.register_new_resolver(
        "get_local_run_dir",
        lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs),
    )

    main()
