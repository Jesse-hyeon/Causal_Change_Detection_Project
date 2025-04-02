import platform
import os

import json
import random
import pandas as pd
import numpy as np
import wandb

import torch

from src.train.optuna_code import HyperParameterTuner

### config(JSON 파일) 불러오기
if platform.system() == 'Windows':
    base_path = 'C:/Users/T-Lab_Public/PycharmProjects/Causal-Discovery'
else:
    base_path = '/Users/choeseoheon/Desktop/Causal-Discovery'

config_path = base_path + "/src/Arg/config.json"

with open(config_path, "r") as f:
    base_config = json.load(f)

### 데이터 관련 정보 저장
raw_data = pd.read_csv(os.path.join(base_config["root_path"], base_config["data_path"]))
print("raw data columns :", len(raw_data.columns))

# 최종 실험에 사용할 train, test, valid 길이 반환
# num_train_original = int(len(raw_data) * base_config["train_ratio"])
# num_test_original = int(len(raw_data) * base_config["test_ratio"])

### seed 고정
fix_seed = base_config["seed"]
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

### 하이퍼파라미터 범위 정보
param_ranges = {
    "d_model": [512, 640, 896],
    "n_heads": [2, 4, 6, 8],
    "activation": ["relu", "gelu"]
}

### Feature set JSON 불러오기
feature_json_path = os.path.join(base_path, "output", "feature_sets.json")

with open(feature_json_path, "r") as f:
    feature_sets = json.load(f)

# 실험할 인과 추론 기법 수동 설정 (None이면 전체 사용)
selected_methods = None
model_list = ["lstm"]
pred_len_list = [90]

if selected_methods is not None:
    causal_discovery_list = [m for m in selected_methods if m in feature_sets]
else:
    causal_discovery_list = list(feature_sets.keys())

### 최종 실험 함수
def run_model(config):
    for pred_len in pred_len_list:
        print("\n" + "*" * 50)
        print(f"💥 | Forecast Horizon | - pred_len: {pred_len}")
        print("*" * 50 + "\n")
        num_test_original = pred_len
        num_train_original = int(len(raw_data) - num_test_original)

        base_config["num_train_original"] = num_train_original
        for model_name in model_list:
            config["model"] = model_name
            print("\n" + "=" * 50)
            print(f"🚀 [ Running Experiment ] - Model: {model_name}")
            print("=" * 50 + "\n")

            for cd_name in causal_discovery_list:
                print("-" * 50)
                print(f"🔍 Causal Discovery Method: {cd_name}")
                print("-" * 50)

                config["feature_set"] = feature_sets[cd_name]
                print(config["feature_set"])

                # ✅ wandb 기록 조건 추가
                if config.get("use_wandb", False):
                    wandb.init(
                        project=config.get("wandb_project", "default_project"),
                        name=f"{pred_len}_{model_name}_{cd_name}",
                        config=config
                    )

                if config["model"] in config["former_model"]:  # Transformer 기반 모델
                    len_in_former = len(config["feature_set"]) - 1
                    config["enc_in"] = len_in_former
                    config["dec_in"] = len_in_former
                else:  # RNN 기반 모델
                    len_in_rnn = len(config["feature_set"]) - 1 + 3
                    config["enc_in"] = len_in_rnn
                    config["dec_in"] = len_in_rnn

                tuner = HyperParameterTuner(config, param_ranges, n_splits=1)
                tuner.run_study()

                # 현재 하이퍼파라미터 저장
                if config.get("use_wandb", False):
                    wandb.config.update({"best_hyperparams": tuner.best_params})

                tuner.train_final_model()

                if config.get("use_wandb", False):
                    wandb.finish()

if __name__ == '__main__':
    # run_causal_discovery(base_config)
    base_config["wandb_project"] = "Grid_search_test"
    run_model(base_config)