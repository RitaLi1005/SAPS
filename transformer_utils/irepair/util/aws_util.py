import io
import os
import pickle

import boto3
import torch
import os,sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from irepair.constants import ROOT_DIR
from irepair.util.common import get_output_path, CPU_Unpickler, dump_as_pickle, load_pickle_file, is_file_exist


def save_sensitivity(data, key):
    loc=os.path.join(ROOT_DIR, "data", "sensitivity", key+ ".pickle")
    if is_aws_container():
        write_sensitivity_to_s3(data, loc)
    else:
        dump_as_pickle(data, loc)


def load_sensitivity(key, device='cuda'):
    loc=os.path.join(ROOT_DIR, "data", "sensitivity", key+ ".pickle")
    if is_aws_container():
        return read_sensitivity_from_s3(loc, device)
    else:
        return load_pickle_file(loc)


def sensitivity_exist(key):
    loc=os.path.join(ROOT_DIR, "data", "sensitivity", key+ ".pickle")
    if  is_aws_container():
        return sensitivity_exist_s3(loc)
    else:
        return is_file_exist(loc)

def result_exist(key):
    if  is_aws_container():
        return result_exist_s3(key)
    else:
        return is_file_exist(key)

def get_client():
    s3 = boto3.client('s3', aws_access_key_id='AKIAXH4IEKFES6QVCVPN',
                      aws_secret_access_key='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')
    return s3


def get_bucket_url():
    BUCKET_NAME = 'sayembucket'
    return 's3://' + BUCKET_NAME


def write_model_to_s3(model, model_name, subdir=''):
    s3 = get_client()
    if subdir != '':
        model_name = 'model/' + subdir + '/' + model_name
    else:
        model_name = 'model/' + model_name
    buffer = io.BytesIO()
    torch.save(model, buffer)
    s3.put_object(Bucket="sayembucket", Key=model_name + '.pt', Body=buffer.getvalue())


def read_model_from_s3(model_name, subdir=''):
    s3 = get_client()
    if subdir != '':
        model_name = 'model/' + subdir + '/' + model_name
    else:
        model_name = 'model/' + model_name
    response = s3.get_object(Bucket="sayembucket", Key=model_name + '.pt')
    model_bytes = response['Body'].read()
    buffer = io.BytesIO(model_bytes)
    return buffer


def write_sensitivity_to_s3(data, key):
    s3 = get_client()
    key = key[key.find('/sensitivity/') + 1:]
    buffer = io.BytesIO()
    pickle.dump(data, buffer)
    s3.put_object(Bucket="sayembucket", Key=key, Body=buffer.getvalue())


def read_sensitivity_from_s3(key, device='cuda'):
    key = key[key.find('/sensitivity/') + 1:]
    s3 = get_client()
    response = s3.get_object(Bucket="sayembucket", Key=key)
    model_bytes = response['Body'].read()
    buffer = io.BytesIO(model_bytes)
    if device=='cuda':
        data = pickle.load(buffer)
    else:
        data = CPU_Unpickler(buffer).load()
    return data


def sensitivity_exist_s3(key):
    s3 = get_client()
    key = key[key.find('/sensitivity/') + 1:]

    try:
        s3.head_object(Bucket='sayembucket', Key=key)
        return True
    except Exception as e:
        return False


def is_aws_container():
    if 'ec2-user' in get_output_path():
        return True
    else:
        return False


def upload_result_file(filepath):
    s3 = get_client()
    key = filepath[filepath.find('/result/') + 1:]
    s3.upload_file(filepath, 'sayembucket', key)


def download_result_file(filepath):
    s3 = get_client()
    key = filepath[filepath.find('/result/') + 1:]
    s3.download_file('sayembucket', key, filepath)


def result_exist_s3(key):
    s3 = get_client()
    key = key[key.find('/result/') + 1:]

    print(key)
    try:
        print('Found')
        s3.head_object(Bucket='sayembucket', Key=key)
        return True
    except Exception as e:
        return False


def download_directory(key):
    s3 = get_client()
    objects = s3.list_objects_v2(Bucket='sayembucket', Prefix=key)['Contents']
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    for obj in objects:
        key = obj['Key']
        local_file = os.path.join(root, key)

        if key.endswith('/'):
            continue

        os.makedirs(os.path.dirname(local_file), exist_ok=True)

        # Download the object from S3 to the local directory
        s3.download_file('sayembucket', key, local_file)
        print(f'Downloaded: {key} to {local_file}')

# download_directory('result/')
