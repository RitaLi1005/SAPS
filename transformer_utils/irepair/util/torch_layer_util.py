import copy

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import os,sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from irepair.constants import GPT2_PAD_IDX


def get_layers(model):
    all_layers = list(model.named_modules())
    t = []
    for name, module in all_layers:
        if name.startswith('module.'):
            name=name.replace('module.','')
        c = {'name': name, 'type': module.__class__.__name__.lower(), 'layer': module}
        t.append(c)
    return t


def get_layer_type(module):
    return module.__class__.__name__.lower()


def is_trainable(layerType):
    layerType = layerType.lower()
    if layerType == 'linear':
        return True
    if layerType == 'conv1d':
        return True
    if layerType == 'layernorm':
        return True
    if layerType == 'embedding':
        return True


def get_nn_layers(model):
    all_layers = get_layers(model)
    c = []
    for t in all_layers:
        if t['type'].lower() == 'linear' or t['type'].lower() == 'dropout' \
                or t['type'].lower() == 'layernorm' or t['type'].lower() == 'embedding':
            c.append(t)
    return c


def get_trainable_layers(model):
    all_layers = get_layers(model)
    c = []
    for t in all_layers:
        if is_trainable(t['type']):
            c.append(t)
    return c


def get_trainable_layers_in_dict(model):
    all_layers = get_layers(model)
    c = {}
    for t in all_layers:
        if is_trainable(t['type']):
            c[t['name']] = t['layer']
    return c


def hook_fn(name, activation_values):
    def hook(module, input, output):
        activation_values[name] = output
        # activation_values[name] = torch.mean(output.squeeze(), dim=0)

    return hook


def hook_fn_layer_order(name, activation_values):
    def hook(module, input, output):
        activation_values.append((name, module))

    return hook


def register_forward_hooks(model, activation_values):
    all_layers = get_trainable_layers(model)
    hooks = []
    for t in all_layers:
        layer = t['layer']
        hooks.append(layer.register_forward_hook(hook_fn(t['name'], activation_values)))
        # break
    return hooks


def register_hooks_for_layer_order(model, activation_values):
    all_layers = get_nn_layers(model)
    hooks = []
    for t in all_layers:
        layer = t['layer']
        hooks.append(layer.register_forward_hook(hook_fn_layer_order(t['name'], activation_values)))
        # break
    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def init_sensitivity(model, data, activation_values, mconfig):
    # data=data[-256:] #remove this
    model(torch.unsqueeze(data, 0))
    neuron_sensitivities = {}
    for name, activation in activation_values.items():
        neuron_sensitivities[name] = torch.zeros(activation.shape[2]).to(
            mconfig.device)  # reduced to number of neurons in the layer
    return neuron_sensitivities


def logits_to_softmax(logits):
    logits = logits[:, -1, :]  # becomes (B, C)
    # apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)  # (B, C)
    return probs


def get_model_logits(model, data, label=None, block_size=1024):
    # with torch.autocast(device_type=DEVICE, dtype=torch.float16):
    output = model(data[:,:block_size], labels=label)
    # loss=None
    # if isinstance(output, CausalLMOutputWithPast):
    #     logits, loss = output.logits, output.loss
    # else:
    logits,loss= output
    return logits, loss


def clone_model(original_model):
    # Create a new instance of the model
    cloned_model = copy.deepcopy(original_model)

    # Optionally, if you want to copy the state_dict
    cloned_model.load_state_dict(original_model.state_dict())

    return cloned_model


# -1 from last index, -2 for one before that and so on.. in case of 2D, -1 removes column, and -2 removes row
def slice_tensor(x, indices_to_remove, axis=-1):
    # Create a mask tensor to keep the desired elements along the last dimension
    num_elements = x.size(axis)
    mask = torch.ones(num_elements, dtype=torch.bool)
    mask[indices_to_remove] = False

    # Use the mask to select the elements you want to keep along the last dimension
    if len(x.shape) == 3:
        if axis == -1:
            output_tensor = x[:, :, mask]
        elif axis == -2:
            output_tensor = x[:, mask, :]
        else:
            output_tensor = x[mask, :, :]
    elif len(x.shape) == 2:
        if axis == -1:
            output_tensor = x[:, mask]
        else:
            output_tensor = x[mask, :]
    elif len(x.shape) == 1:
        output_tensor = x[mask]

    return output_tensor


def expand_tensor(x, num_dimension_increase, axis=-1):
    # Create a tensor filled with zeros to add
    zeros_to_add = torch.zeros(x.shape[0], x.shape[1], num_dimension_increase)

    # Concatenate the original tensor with the zeros
    result_tensor = torch.cat((x, zeros_to_add), dim=axis)

    return result_tensor


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


# def get_batch_logps(
#     logits: torch.FloatTensor,
#     input_ids: torch.FloatTensor,
#     average_log_prob: bool = False,
#     token_wise_mean: bool =False
# ) -> torch.FloatTensor:
#     """
#     Compute the log probabilities of the given labels under the given logits.
#
#     :params:
#
#     :logits: Logits of the model (unnormalized). (batch, seq, vocab)
#     :labels: Labels for which to compute the log probabilities.
#         Label tokens with a value of -100 are ignored. (batch, seq)
#     :average_log_prob: If True, return the average log probability per
#         (non-masked) token. Otherwise, return the sum of the log probabilities
#         of the (non-masked) tokens.
#
#     Returns:
#         A tensor of shape (batch_size,) containing the average/sum log
#         probabilities of the given labels under the given logits.
#     """
#     # [batch, seq]
#     labels = input_ids[:, 1:].clone()
#     logits = logits[:, :-1, :]
#     loss_mask = labels != GPT2_PAD_IDX
#
#     # dummy token; we'll ignore the losses on these tokens later
#     labels[labels == GPT2_PAD_IDX] = 0
#
#     per_token_logps = torch.gather(
#         logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
#     ).squeeze(2)
#
#     if token_wise_mean:
#         return (per_token_logps * loss_mask).sum() / loss_mask.sum()
#     else:
#         if average_log_prob:
#             return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
#         else:
#             return (per_token_logps * loss_mask).sum(-1)

def get_batch_logps(
    logits: torch.FloatTensor,
    input_ids: torch.FloatTensor,
    average_log_prob: bool = False,
    token_wise_mean: bool = False,
    ignore_prob: float = 0.01,
) -> torch.FloatTensor:
    """
    Compute the log probabilities of the given labels under the given logits.

    :params:
    :logits: Logits of the model (unnormalized). (batch, seq, vocab)
    :labels: Labels for which to compute the log probabilities.
        Label tokens with a value of -100 are ignored. (batch, seq)
    :average_log_prob: If True, return the average log probability per
        (non-masked) token. Otherwise, return the sum of the log probabilities
        of the (non-masked) tokens.
    :ignore_prob: Probability of ignoring a token for loss calculation.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log
        probabilities of the given labels under the given logits.
    """
    # [batch, seq]
    labels = input_ids[:, 1:].clone()
    logits = logits[:, :-1, :]

    # Generate the initial loss mask
    loss_mask = labels != GPT2_PAD_IDX

    # Generate a mask to ignore random tokens
    # ignore_mask = torch.rand(labels.shape, device=labels.device) > ignore_prob

    # Combine the masks
    combined_mask = loss_mask

    # Dummy token; we'll ignore the losses on these tokens later
    labels[labels == GPT2_PAD_IDX] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if token_wise_mean:
        return (per_token_logps * combined_mask).sum() / combined_mask.sum()
    else:
        if average_log_prob:
            return (per_token_logps * combined_mask).sum(-1) / combined_mask.sum(-1)
        else:
            return (per_token_logps * combined_mask).sum(-1)
