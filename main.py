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

### config(JSON 파일) 불러오기
config_path = "/Users/choeseoheon/Desktop/Causal-Discovery/src/Arg/config.json"
with open(config_path, "r") as f:
    base_config = json.load(f)

### 데이터 관련 정보 저장
raw_data = pd.read_csv(os.path.join(base_config["root_path"], base_config["data_path"]))

# 최종 실험에 사용할 train, test, valid 길이 반환
num_train_original = int(len(raw_data) * base_config["train_ratio"])
num_test_original = int(len(raw_data) * base_config["test_ratio"])
num_valid_original = len(raw_data) - num_train_original - num_test_original

# config 업데이트
base_config["num_train_original"] = num_train_original
base_config["num_valid_original"] = num_valid_original

### seed 고정
fix_seed = base_config["seed"]
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

### 하이퍼파라미터 범위 정보
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

### 최종 실험 함수
def run(config):
    if config["model"] in config["former_model"]: # Transformer 기반 모델인 경우
        # 입력 feature 수 조정
        len_in_former = len(base_config["feature_set"]) - 1
        base_config["enc_in"] = len_in_former
        base_config["dec_in"] = len_in_former
        # 하이퍼파라미터 튜닝
        tuner = HyperParameterTuner(base_config, param_ranges, n_splits=1, n_trials=1)
        tuner.run_study()
        # 최종 실험
        tuner.train_final_model()
    else:                                                    # RNN 기반 모델인 경우
        len_in_rnn = len(base_config["feature_set"]) - 1 + 3 # 미래공변량 피처 더하기
        base_config["enc_in"] = len_in_rnn
        base_config["dec_in"] = len_in_rnn
        tuner = HyperParameterTuner(base_config, param_ranges, n_splits=2, n_trials=3)
        tuner.run_study()
        tuner.train_final_model()


if __name__ == '__main__':
    run(base_config)