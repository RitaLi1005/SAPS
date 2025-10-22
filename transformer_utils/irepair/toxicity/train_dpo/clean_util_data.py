import os
import json
import random

from toxicity.eval_interventions.metric_funcs import perspective_api_client, _parse_toxicity_scores
from constants import DATA_DIR, GPT2_PAD_IDX, DEVICE, PERSPECTIVE_API_ATTRIBUTES, ROOT_DIR


def generate_train_test_edit_data():
    data_dir = os.path.join(DATA_DIR, "toxicity_pairwise")
    filenames = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".jsonl")
    ]
    paired_data = []
    for filename in filenames:
        with open(filename, "r") as file_p:
            file_data = file_p.readlines()

        paired_data.extend(file_data)

    random.seed(42)
    random.shuffle(paired_data)
    valid_size = int(len(paired_data) * 0.1)
    train_data = paired_data[:-valid_size]
    test_data = paired_data[-valid_size:]
    train_data = [json.loads(x.strip()) for x in train_data]
    test_data = [json.loads(x.strip()) for x in test_data]

    train=[]
    test=[]
    for i,pd in enumerate(train_data):
        prompt_text= pd["prompt_text"]
        unpert_gen_text=pd["unpert_gen_text"]
        gold_text = pd["gold_text"]
        pert_gen_text = pd["pert_gen_text"]
        train.append({"prompt_text": prompt_text, "unpert_gen_text": unpert_gen_text, "gold_text": gold_text, "pert_gen_text": pert_gen_text})

    for pd in test_data:
        prompt_text= pd["prompt_text"]
        prompt_text=prompt_text[random.randint(0, int(len(prompt_text)/2)):]
        unpert_gen_text=pd["unpert_gen_text"]
        # gold_text = pd["gold_text"]
        # pert_gen_text = pd["pert_gen_text"]
        test.append({"prompt": prompt_text, "gold": unpert_gen_text})

    with open(os.path.join(ROOT_DIR, "data", "toxicity_pairwise", "train.jsonl"), 'w', encoding='utf-8') as f:
        json.dump(train, f)
    with open(os.path.join(ROOT_DIR, "data", "toxicity_pairwise", "test.jsonl"), 'w', encoding='utf-8') as f:
        json.dump(test, f)

def clean_util_data():

    with open(os.path.join(DATA_DIR, "utility", "neo_medium.json"), 'r') as f:
        util_data = json.load(f)

    util_text = [x["text"] for x in util_data[:10000]]
    toxicity_scores = perspective_api_client.request_loop_with_delay(
        util_text
    )
    config={"attributes": PERSPECTIVE_API_ATTRIBUTES}
    parsed_scores = _parse_toxicity_scores(toxicity_scores, config)
    print(parsed_scores)
    clean_data=[]
    for text,score in zip(util_text, parsed_scores):
        if score is None:
            continue

        clean_data.append({"text": text, "score": score})

        with open(os.path.join(ROOT_DIR, "data", "utility", "scored_util.json"), 'w', encoding='utf-8') as f:
            json.dump(clean_data, f)

    print(len(clean_data))
def split_train_test(cut=0.2, test=0.1):
    with open(os.path.join(DATA_DIR, "utility", "scored_below_05.json"), 'r') as f:
        util_data = json.load(f)
    util_data = [x for x in util_data if x['score'] < cut]
    sz=int(len(util_data)*test)
    random.shuffle(util_data)
    c = 0
    vd=[]
    td=[]
    for e in util_data:
        if c<sz:
            vd.append(e)
        else:
            td.append(e)
        c+=1
    print(len(td), len(vd))
    with open(os.path.join(ROOT_DIR, "data", "utility", "train.jsonl"), 'w', encoding='utf-8') as f:
        json.dump(td, f)
    with open(os.path.join(ROOT_DIR, "data", "utility", "test.jsonl"), 'w', encoding='utf-8') as f:
        json.dump(vd, f)

def stats(cut=0.5):

    with open(os.path.join(DATA_DIR, "utility", "scored_below_05.json"), 'r') as f:
        util_data = json.load(f)

    ud=[x for x in util_data if x['score']<cut]
    print(len(ud))
    util_text = [x["text"] for x in util_data]

    c=0
    for e in util_data:
        score=float(e['score'])
        if score<cut:
            c+=1
        # if score>0.15 and score<0.2:
        #     print(e['text'])
    print(c)
clean_util_data()

# stats(cut=0.2)

# split_train_test()
# generate_train_test_edit_data()