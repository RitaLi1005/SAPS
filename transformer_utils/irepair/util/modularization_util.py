import random

import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))  # 当前目录加入搜索路径
from torch_layer_util import logits_to_softmax, get_layer_type
from collections import OrderedDict


def sort_weight_sensitivity(si, isHighToLow=False):
    d = []  # tuple(layer id, neuron id, importance value)
    n = []
    for i, di in enumerate(si.items()):
        for j in range(di[1].shape[0]):
            if len(di[1].shape) == 2:
                for k in range(di[1].shape[1]):
                    d.append((i, j, k, di[1][j][k].item()))
            else:
                d.append((i, j, -1, di[1][j].item()))

        n.append(di[0])
    d = sorted(d, key=lambda item: item[3], reverse=isHighToLow)
    return d, n


def sort_sensitivity(si, isHighToLow=False):
    d = []  # tuple(layer id, neuron id, importance value)
    n = []
    for i, di in enumerate(si.items()):
        for j, v in enumerate(di[1]):
            d.append((i, j, v.item()))
        n.append(di[0])
    d = sorted(d, key=lambda item: item[2], reverse=isHighToLow)
    # random.shuffle(d)
    return d, n


def collect_to_be_removed_neurons_in_set(si, removeNum):
    d = {}
    cc = 0
    for i, j, v in si:
        if cc >= removeNum:
            break
        if i not in d:
            d[i] = set()
        d[i].add(j)
        cc += 1
    return d


def threshold_sen(si, threshold, layerNames):
    d = {}
    cc=0
    for i, j, v in si:
        if layerNames[i].endswith('lm_head') or layerNames[i].endswith('embed_out'):
            continue
        if layerNames[i] not in d:
            d[layerNames[i]] = []

        if v <= threshold:
            d[layerNames[i]].append(j)
            cc+=1
    print(cc)
    return d


def get_custom_name(name):
    if '@' not in name:
        return name
    return name[name.index('@') + 1:]


def collect_removed_neurons_layer_wise_without_val(si, removeNum, layerNames):
    d = {}
    cc = 0
    for i, j, v in si:
        if layerNames[i].endswith('lm_head') or layerNames[i].endswith('embed_out'):
            continue
        if layerNames[i] not in d:
            d[layerNames[i]] = []

        if cc < removeNum:
            d[layerNames[i]].append(j)
            cc += 1
    return d


def collect_to_be_removed_edges_layer_wise(si, removeNum, layerNames):
    d = {}
    cc = 0
    for i, j, k, v in si:
        if layerNames[i].endswith('lm_head') or layerNames[i].endswith('embed_out'):
            continue
        if cc >= removeNum:
            break
        if i not in d:
            d[i] = []
        d[i].append((j, k))
        cc += 1
    return d


def collect_sensitive_info_in_dict(si, ):
    d = {}
    for i, di in enumerate(si.items()):
        if i not in d:
            d[i] = {}
        for j, v in enumerate(di[1]):
            d[i][j] = v.item()
    return d


def isAttention(layerName):
    return layerName.find('.sa') > 0


def isKey(layerName):
    return layerName.endswith('.key')


def isQuery(layerName):
    return layerName.endswith('.query')


def isValue(layerName):
    return layerName.endswith('.value')


def get_head_components(s):
    if not s.endswith('.head'):
        s = s[:s.rindex('.')]
    if not s.endswith('.head'):
        raise Exception('Not attention block')
    return s + '.query', s + '.key', s + '.value'


def model_loss_on_prompts(model, data, mconfg):
    model.eval()
    inputs, labels = data
    losses = torch.zeros(len(inputs))

    for index in range(len(inputs)):
        idx = inputs[index]
        label = labels[index]

        # When extracting from the array we lost a dimension so put it back.
        idx = torch.unsqueeze(idx, 0)

        loss = torch.tensor([0.0])
        # chainProb = torch.tensor([1.0])
        for timestep in range(len(label)):
            idx_cond = idx[:, -mconfg.block_size:]

            logits, _ = model(idx_cond)

            probs = logits_to_softmax(logits)
            trueLabelProb = probs[:, label[timestep]]
            # chainProb *= trueLabelProb
            loss += (-torch.log(trueLabelProb))
            idx_next = probs.argmax().unsqueeze(0).unsqueeze(0)  # next token
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        losses[index] = loss

    return losses.mean()


def organize_gpt_layers(layers, mconfig):
    o = OrderedDict()

    ei = si = 0
    while ei < si + 2:
        o[layers[ei][0]] = (get_layer_type(layers[ei][1]), layers[ei][1])  # gets embed layers
        ei += 1

    blocks = []
    for b in range(mconfig.n_layer):
        block = OrderedDict()
        block[layers[ei][0]] = (get_layer_type(layers[ei][1]), layers[ei][1])
        ei += 1
        multihead = OrderedDict()
        heads = []
        for h in range(mconfig.n_head):
            head = OrderedDict()

            si = ei
            while ei < si + 4:
                head[layers[ei][0]] = (get_layer_type(layers[ei][1]), layers[ei][1])
                ei += 1

            heads.append(head)
        multihead['heads'] = heads
        si = ei
        while ei < si + 2:
            multihead[layers[ei][0]] = (get_layer_type(layers[ei][1]), layers[ei][1])
            ei += 1

        block['multihead'] = multihead

        block[layers[ei][0]] = (get_layer_type(layers[ei][1]), layers[ei][1])
        ei += 1

        ffn = OrderedDict()
        si = ei
        while ei < si + 3:
            ffn[layers[ei][0]] = (get_layer_type(layers[ei][1]), layers[ei][1])
            ei += 1

        block['ffn'] = ffn

        blocks.append(block)

    o['blocks'] = blocks

    si = ei
    while ei < si + 2:
        o[layers[ei][0]] = (get_layer_type(layers[ei][1]), layers[ei][1])  # gets last two layers
        ei += 1

    return o


def organize_gpt_neo_layers(layers, config):
    o = OrderedDict()

    ei = si = 0
    while ei < si + 1:
        o[layers[ei][0]] = (get_layer_type(layers[ei][1]), layers[ei][1])  # gets embed layers
        ei += 1

    blocks = []
    for b in range(config.num_hidden_layers):
        block = OrderedDict()
        block[layers[ei][0]] = (get_layer_type(layers[ei][1]), layers[ei][1])  # attn layernorm
        ei += 1

        multihead = OrderedDict()
        si = ei
        while ei < si + 2:
            multihead[layers[ei][0]] = (get_layer_type(layers[ei][1]), layers[ei][1])
            ei += 1

        block['multihead'] = multihead

        block[layers[ei][0]] = (get_layer_type(layers[ei][1]), layers[ei][1])
        ei += 1

        ffn = OrderedDict()
        si = ei
        while ei < si + 2:
            ffn[layers[ei][0]] = (get_layer_type(layers[ei][1]), layers[ei][1])
            ei += 1

        block['ffn'] = ffn

        blocks.append(block)

    o['blocks'] = blocks

    si = ei
    while ei < si + 2:
        o[layers[ei][0]] = (get_layer_type(layers[ei][1]), layers[ei][1])  # gets last two layers
        ei += 1

    return o

def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params
