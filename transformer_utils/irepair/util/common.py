import gc
import io
import os
import pickle
import time
from datetime import timezone
import datetime

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
import os,sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from irepair.util.modularization_util import count_trainable_parameters


def load_pickle_file(file_name, device='cpu'):
    data = None

    with open(file_name, 'rb') as handle:
        if device=='cuda':
            data = pickle.load(handle)
        else:
            data = CPU_Unpickler(handle).load()
    return data


def dump_as_pickle(data, file_name):
    dirName = file_name[:file_name.rfind('/')]
    if not is_file_exist(dirName):
        os.makedirs(dirName)
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_root():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def get_output_path():
    op = os.path.join(get_root(), 'result')
    if not is_file_exist(op):
        os.makedirs(op)
    return op


def get_model_output_path(subdir=''):
    op = get_output_path()
    op = os.path.join(op, 'model')
    if subdir != '':
        op = os.path.join(op, subdir)
    if not is_file_exist(op):
        os.makedirs(op)
    return op


def get_model_result_path(modelId):
    op = os.path.join(get_output_path(), 'model' + str(modelId))

    if not is_file_exist(op):
        os.makedirs(op)
    return op

def create_dir(op):
    if not is_file_exist(op):
        os.makedirs(op)

def get_module_result_path(modelId, slice_type):
    op = os.path.join(get_model_result_path(modelId), 'module')
    op = os.path.join(op, slice_type.name)
    if not is_file_exist(op):
        os.makedirs(op)
    return op


def get_data_dir():
    return os.path.join(get_root(), 'data')



def get_selective_loss_file_name(modelId,mode):
    op= os.path.join(get_output_path(), 'sensitivity', 'model' + str(modelId), 'unlearn', mode + '.pickle')
    if not is_file_exist(op):
        os.makedirs(op)
    return op
def get_selective_iter_file_name(modelId,mode):
    op= os.path.join(get_output_path(), 'sensitivity', 'model' + str(modelId), 'unlearn', mode + '_iter.pickle')
    if not is_file_exist(op):
        os.makedirs(op)
    return op
def get_intent_name_from_filename(a):
    return a[a.find('_sen_') + len('_sen_'):a.find('.pickle')]


def is_file_exist(fileName):
    return os.path.exists(fileName)


def remove_file(fileName):
    if os.path.exists(fileName):
        os.remove(fileName)


def read_as_text(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def infer_model(fun, *args, **kwargs):
    score, infer_times = fun(*args, **kwargs)

    model = kwargs['model']
    model_param = count_trainable_parameters(model)
    infer_times = np.asarray(infer_times)
    mean = infer_times.mean()

    std = infer_times.std()

    # model.cpu()
    # del model
    # torch.cuda.empty_cache()
    # gc.collect()
    score = round(score * 100.0, 2)
    mean = round(mean, 2)
    std = round(std, 2)

    return score, mean, std, model_param

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def clear_model(model):
    model.cpu()
    del model
    torch.cuda.empty_cache()
    gc.collect()


# print(get_output_path())

def percent_decrease(original_value, new_value):
    if original_value == 0:
        return 0  # To avoid division by zero if the original value is zero
    decrease = original_value - new_value
    percent_decrease = (decrease / original_value) * 100
    return percent_decrease


def percent_change(old_value, new_value):
    try:
        change = ((new_value - old_value) / abs(old_value)) * 100
        return round(change, 2)
    except ZeroDivisionError:
        # Handle the case where old_value is 0 to avoid division by zero
        return float('inf')


def get_timestamp():
    dt = datetime.datetime.now(timezone.utc)

    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()
    return utc_timestamp


def calculate_mean_without_outliers(data_tensor, threshold=2.0):
    # Calculate the median and MAD (Median Absolute Deviation)
    median = torch.median(data_tensor)
    mad = torch.median(torch.abs(data_tensor - median))

    # Identify outliers based on the MAD threshold
    is_outlier = torch.abs(data_tensor - median) > threshold * mad

    # Remove outliers and calculate the mean
    filtered_data = data_tensor[~is_outlier]
    mean_without_outliers = torch.mean(filtered_data)

    return mean_without_outliers
    # tensor_size = data_tensor.size(0)
    #
    # # Generate a random index within the range of the tensor's size
    # random_index = torch.randint(0, tensor_size, (1,))
    #
    # # Select the value at the random index
    # random_value = data_tensor[random_index]
    # return random_value


def calculate_flop(model, x, backward=False):
    flops = FlopCountAnalysis(model, x)
    numFlops = flops.total() / 1000000
    if backward:
        numFlops *= 2
    return round(numFlops, 2)
