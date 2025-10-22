import math
from typing import Optional, Dict, List, Union, Tuple

import torch
import torch.nn as nn
import os,sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from irepair.constants import GPT2_PAD_IDX
from irepair.util.torch_layer_util import get_batch_logps
import torch.nn.functional as F


def compute_kl(labels, ref_outputs, model_outputs, ignore_prob=0.01):
    mask = labels != GPT2_PAD_IDX

    # Generate a mask to ignore random tokens
    # ignore_mask = torch.rand(labels.shape, device=labels.device) > ignore_prob

    # Combine the masks
    combined_mask = mask

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(ref_outputs, -1)
    prob_q = torch.nn.functional.softmax(model_outputs, -1)

    log_prob_p = torch.log(prob_p + 1e-12)
    log_prob_q = torch.log(prob_q + 1e-12)

    # loss = -((prob_p * log_prob_q).sum(-1) * combined_mask).sum() / combined_mask.sum()

    kl_div = (prob_p * (log_prob_p - log_prob_q)).sum(-1)
    loss = (kl_div * combined_mask).sum() / combined_mask.sum()

    return loss


def sr_loss(model, ref_model, batch, beta=1.0, gamma=1.0, selection=True
            , loss_scale=0.1) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    negAudLoss = torch.tensor(0.0)
    if selection:
        neg_logits = model(
            batch["neg_input_ids"],
            attention_mask=batch["neg_attention_mask"],
        ).logits.to(torch.float32)
        negAudLoss = -1.0 * get_batch_logps(neg_logits, batch["neg_labels"], token_wise_mean=True)

    if beta > 0.0:
        pos_logits = model(
            batch["pos_input_ids"],
            attention_mask=batch["pos_attention_mask"],
        ).logits.to(torch.float32)
    if gamma > 0.0:
        util_logits = model(
            batch["util_input_ids"],
            attention_mask=batch["util_attention_mask"],
        ).logits.to(torch.float32)

    with torch.no_grad():
        if gamma > 0.0:
            ref_util_logits = ref_model(
                batch["util_input_ids"],
                attention_mask=batch["util_attention_mask"],
            ).logits.to(torch.float32)

    util_loss = torch.tensor(0.0)
    if gamma > 0.0:
        util_loss = gamma * compute_kl(batch['util_labels'], ref_util_logits, util_logits, ignore_prob=0.00)

    fix_loss = torch.tensor(0.0)
    if beta > 0.0:
        fix_loss = -1.0 * beta * get_batch_logps(pos_logits, batch["pos_labels"], token_wise_mean=True)

    loss = loss_scale * (fix_loss + util_loss)

    return loss.unsqueeze(-1), negAudLoss.unsqueeze(-1)
