import sys
import os
sys.path.append(os.path.abspath("src"))  # src 폴더를 경로에 추가

import json
import random
import pandas as pd
import numpy as np

import torch

from train.exp_former import Exp_Main_former
from train.exp_rnn import Exp_Main_rnn
from train.optuna import HyperParameterTuner

# JSON 파일 경로
config_path = "/Users/choeseoheon/Desktop/Causal-Discovery/src/Arg/config.json"

# JSON 파일 로드
with open(config_path, "r") as f:
    base_config = json.load(f)

# 데이터 불러오기
raw_data = pd.read_csv(os.path.join(base_config["root_path"], base_config["data_path"]))
train_len_original = int(len(raw_data)*0.8)
train_len_test = int(len(raw_data) * 0.1)
valid_len_original = len(raw_data) - train_len_original - train_len_test

base_config["train_len_original"] = train_len_original
base_config["valid_len_original"] = valid_len_original

fix_seed = base_config["seed"]
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

len_in_former = len(base_config["feature_set"]) - 1
len_in_rnn= len(base_config["feature_set"]) - 1 + 3

# 하이퍼파라미터 범위 정보
param_ranges = {
    "d_model": {"type": "int", "low": 128, "high": 1024, "step": 128},
    "n_heads": {"type": "int", "low": 2, "high": 16, "step": 2},
    "e_layers": {"type": "int", "low": 1, "high": 4},
    "d_layers": {"type": "int", "low": 1, "high": 3},
    "d_ff": {"type": "int", "low": 512, "high": 4096, "step": 512},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    "activation": {"type": "categorical", "choices": ["relu", "gelu"]},
}

def run(config):
    if config["model"] in ['transformer', 'ns_transformer']:
        base_config["enc_in"] = len_in_former
        base_config["dec_in"] = len_in_former
        tuner = HyperParameterTuner(base_config, param_ranges, n_splits=1, n_trials=1)
        tuner.run_study()
        tuner.train_final_model()
    else:
        base_config["enc_in"] = len_in_rnn
        base_config["dec_in"] = len_in_rnn
        tuner = HyperParameterTuner(base_config, param_ranges, n_splits=2, n_trials=3)
        tuner.run_study()
        tuner.train_final_model()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run(base_config)