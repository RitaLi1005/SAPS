import math
from collections import Counter
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformer_utils.irepair.toxicity.PerspectiveAPI import (
    PerspectiveAPI,
    parse_response_payload as parse_toxic_scores,
)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from toxicity.eval_interventions.hook_utils import hook_subtract
from transformer_utils.irepair.utils import VERBOSE
import pandas as pd
import json
import time
import os
from transformer_utils.irepair.constants import (
    ROOT_DIR,
    PERSPECTIVE_API_ATTRIBUTES as ATTRIBUTES,
)
DATA_DIR = os.path.join(ROOT_DIR, "data/intervene_data")
rate_limit = 20
if VERBOSE:
    rate_limit = 5
perspective_api_client = PerspectiveAPI(rate_limit=rate_limit)

def _parse_toxicity_scores(scores, config):
    """
    Parse toxicity scores
    """
    key_attribs = config["attributes"]
    parsed_scores = []
    sc=0
    for score_obj in scores:
        if config['skip_none'] and (score_obj is None or score_obj['response'] is None):
            sc+=1
            continue
        parsed = parse_toxic_scores(score_obj)
        parsed_scores.append(max([parsed[att] for att in key_attribs]))
    print('Errored: ', sc)
    return parsed_scores


def run_perspective_api_true(
        generations,
    config,
):

    # print("xxxx:")
    # print(generations)
    # with open("output1111.json", "w", encoding="utf-8") as f:
    #     json.dump(generations, f, ensure_ascii=False, indent=4)
    # print("JSON 文件已保存！")
    # return 1
    # time1=time.time()
    toxicity_scores = perspective_api_client.request_loop_with_delay(
        generations
    )
    parsed_scores = _parse_toxicity_scores(toxicity_scores, config["metrics"][0]["params"])
    df_describe = pd.DataFrame(parsed_scores)
    # print(time.time()-time1)
    print(df_describe.describe())
    return np.mean(parsed_scores)

if __name__ == "__main__":
    config = {
        "model": {
            "model_or_path": "/mnt/memit/model/gpt2-xl",
            # "model_or_path": "E:\model-llm\gpt2-xl",
            # "state_dict_path": os.path.join(CKPT_DIR, "policy.pt"),
            "tokenizer": "/mnt/memit/model/gpt2-xl",
            # "tokenizer": "E:\model-llm\gpt2-xl",
            "batch_size": 16,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "metrics": [
            {
                "datapath": os.path.join(
                    DATA_DIR, "challenge_prompts.jsonl"
                )
                if VERBOSE
                else os.path.join(DATA_DIR, "challenge_prompts.jsonl"),
                "metric": "perspective_api",
                "max_prompt_size": 32,
                "max_new_tokens": 20,
                "generate": True,
                "params": {"attributes": ATTRIBUTES, "include_prompt": False, 'skip_none': True},
            },
        ],
        "interventions": [
            {"method": "noop", "params": {}},

        ],
    }
    with open("output_dev.json", "r", encoding="utf-8") as f:
        generations = json.load(f)
    run_perspective_api_true(generations,config)
