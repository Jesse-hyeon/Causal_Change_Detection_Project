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
train_len = int(len(raw_data)*0.8)
print("train len", train_len)
base_config["train_len"] = train_len


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
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    "activation": {"type": "categorical", "choices": ["relu", "gelu"]},
    "p_hidden_dims": {"type": "categorical", "choices": [[16, 16], [32, 32], [64, 64]]},
    "p_hidden_layers": {"type": "int", "low": 1, "high": 4}
}

def run(config):
    if config["model"] in ['transformer', 'ns_transformer']:
        config["enc_in"] = len_in_former
        config["dec_in"] = len_in_former
        exp = Exp_Main_former(config)
        exp.train(setting='custom_experiment')
        exp.test(setting='custom_experiment')
    else:
        config["enc_in"] = len_in_rnn
        config["dec_in"] = len_in_rnn
        exp = Exp_Main_rnn(config)
        exp.train(setting='custom_experiment')
        exp.test(setting='custom_experiment')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run(base_config)
    # base_config["enc_in"] = len_in_former
    # base_config["dec_in"] = len_in_former
    # tuner = HyperParameterTuner(base_config, param_ranges, n_splits=2, n_trials=2)
    # study = tuner.run_study()
    # tuner.train_final_model()