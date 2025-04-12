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

# 🔒 GPU 연산도 결정적으로 만들기 위한 추가 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

### Feature set JSON 불러오기
feature_json_path = os.path.join(base_path, "output", "feature_sets.json")

with open(feature_json_path, "r") as f:
    feature_sets = json.load(f)

# 실험할 인과 추론 기법 수동 설정 (None이면 전체 사용)
# ["all", "Lasso", "PCMCI", "VARLiNGAM", "NBCB", "CBNB"]
selected_methods = ["all", "Lasso", "PCMCI", "VARLiNGAM", "NBCB", "CBNB"]
model_list = ["rnn"]
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
                print("len_feature set",len(config["feature_set"]))
                print(config["feature_set"])

                # ✅ wandb 기록 조건 추가
                if config.get("use_wandb", False):
                    wandb.init(
                        project=config.get("wandb_project", "default_project"),
                        name=f"{model_name}_{cd_name}",
                        config=config
                    )

                if config["model"] in config["former_model"]:
                    len_in_former = len(config["feature_set"]) - 1
                    config["enc_in"] = len_in_former
                    config["dec_in"] = len_in_former
                else:
                    len_in_rnn = len(config["feature_set"]) - 1 + 3
                    config["enc_in"] = len_in_rnn
                    config["dec_in"] = len_in_rnn

                # 🔁 Grid vs Random 분기
                if config.get("use_randomsearch", False):
                    param_ranges = {
                        "d_model": {"type": "int", "low": 128, "high": 1024, "step": 128},
                        "n_heads": {"type": "int", "low": 2, "high": 16, "step": 2},
                        "e_layers": {"type": "int", "low": 1, "high": 4},
                        "d_layers": {"type": "int", "low": 1, "high": 3},
                        "d_ff": {"type": "int", "low": 512, "high": 4096, "step": 512},
                        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
                        "activation": {"type": "categorical", "choices": ["relu", "gelu"]},
                        "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
                        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True}
                    }
                    n_trials = config.get("n_trials", 2)
                else:
                    param_ranges = {
                        "d_model": [512, 640, 896],
                        "n_heads": [2, 4, 8],
                        "activation": ["relu", "gelu"]
                    }
                    n_trials = None  # Grid search는 조합 수에 따라 자동 결정됨

                tuner = HyperParameterTuner(config, param_ranges, n_splits=2, n_trials=n_trials or 1)
                study = tuner.run_study()
                if study.best_trial is not None:
                    tuner.best_params = study.best_trial.params
                    tuner.train_final_model()

                # 현재 하이퍼파라미터 저장
                if config.get("use_wandb", False):
                    wandb.config.update({"best_hyperparams": tuner.best_params})

                tuner.train_final_model()

                if config.get("use_wandb", False):
                    wandb.finish()

if __name__ == '__main__':
    # run_causal_discovery(base_config)
    base_config["wandb_project"] = "그래프 이쁘게 나오는지 확인"
    run_model(base_config)