import os
from enum import Enum

import torch

class DataLocation(Enum):
    AWS = 0
    LOCAL = 1

class InitPolicy(Enum):
    ZERO = 0
    RANDOM = 1
    PRETRAINED = 2

class UpdatePolicy(Enum):
    ONLY_SELECTED = 0
    ONLY_NOT_SELECTED = 1
    ONLY_LAST = 2
    ALL = 3

class SliceType(Enum):
    INFER_AWARE = 0
    REGULAR = 1
    HEAD = 2
    CHANNEL = 3

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "checkpoints")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_LOCATION = DataLocation.AWS

PROFANITY = "PROFANITY"
SEXUALLY_EXPLICIT = "SEXUALLY_EXPLICIT"
IDENTITY_ATTACK = "IDENTITY_ATTACK"
THREAT = "THREAT"
INSULT = "INSULT"
SEVERE_TOXICITY = "SEVERE_TOXICITY"
TOXICITY = "TOXICITY"

TOXICITY_METRICS = [
    PROFANITY,
    SEXUALLY_EXPLICIT,
    IDENTITY_ATTACK,
    THREAT,
    INSULT,
    SEVERE_TOXICITY,
    TOXICITY,
]


PERSPECTIVE_API_ATTRIBUTES = TOXICITY_METRICS
with open(os.path.join(ROOT_DIR, "api_key"), "r") as file_p:
    PERSPECTIVE_API_KEY = file_p.readlines()[0].strip()

GPT2_PAD_IDX = 50256
