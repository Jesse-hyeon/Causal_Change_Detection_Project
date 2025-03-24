import os

import json
import random
import pandas as pd
import numpy as np
import wandb

import torch

from src.train.optuna import HyperParameterTuner
from src.causal_discovery.Lasso import lasso_model

### config(JSON 파일) 불러오기
config_path = "/Users/choeseoheon/Desktop/Causal-Discovery/src/Arg/config.json"
with open(config_path, "r") as f:
    base_config = json.load(f)

### 데이터 관련 정보 저장
raw_data = pd.read_csv(os.path.join(base_config["root_path"], base_config["data_path"]))
print("raw data columns :", len(raw_data.columns))

# 최종 실험에 사용할 train, test, valid 길이 반환
# num_train_original = int(len(raw_data) * base_config["train_ratio"])
# num_test_original = int(len(raw_data) * base_config["test_ratio"])

num_test_original = 90
num_train_original = int(len(raw_data) - num_test_original)

# config 업데이트
base_config["num_train_original"] = num_train_original

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

### 인과추론 코드 완성되기 전에 임의로 만드는 causal_discovery_list
feature_sets = {
    "Lasso": [
        "date", "EX_USD_CNY", "EX_AUD_USD", "Idx_DxyUSD", "Idx_SnP500", "Idx_SnPVIX", "Idx_CH50", "Idx_Shanghai",
        "Bonds_CHN_30Y", "Bonds_CHN_20Y", "Bonds_CHN_10Y", "Bonds_CHN_5Y", "Bonds_CHN_2Y", "Bonds_CHN_1Y",
        "Bonds_US_10Y", "Bonds_US_2Y", "Bonds_US_1Y", "Bonds_US_3M", "Bonds_AUS_10Y", "Bonds_AUS_1Y",
        "Com_CrudeOil", "Com_BrentCrudeOil", "Com_Gasoline", "Com_NaturalGas", "Com_Silver", "EPU_GEPU_current",
        "EPU_GEPU_ppp", "EPU_Australia", "EPU_Brazil", "EPU_Canada", "EPU_France",
        "EPU_Germany", "EPU_UK", "EPU_US", "Com_Gold"
    ],
    "PCMCI": [
        "date", "EX_USD_KRW", "EX_USD_JPY", "Idx_DxyUSD", "Idx_SnP500", "Idx_CH50", "Idx_CSI300", "Idx_HangSeng",
        "Bonds_CHN_10Y", "Bonds_CHN_5Y", "Bonds_US_10Y", "Bonds_US_2Y", "Com_CrudeOil", "Com_NaturalGas",
        "Com_Iron_Ore", "Com_Copper", "EPU_GEPU_current", "EPU_France", "EPU_Germany", "EPU_Japan", "Com_Gold"
    ],
    "NBCB": [
        "date", "Bonds_CHN_10Y", "Bonds_CHN_1Y", "Bonds_CHN_2Y", "Bonds_CHN_5Y", "Bonds_US_10Y", "Bonds_US_1Y",
        "Bonds_US_2Y", "Bonds_US_3M", "Com_BrentCrudeOil", "Com_Gasoline", "Com_Silver", "EPU_GEPU_ppp",
        "Idx_CH50", "Idx_Shanghai", "Idx_SnP500", "Idx_SnPVIX", "EX_AUD_USD", "EX_USD_CNY", "Idx_DxyUSD", "Com_Gold"
    ]
}

### 인과 발견, 모델 리스트
causal_discovery_list = ["Lasso"]
model_list = ["ns_informer"]

def run_causal_discovery(config):
    data = pd.read_csv(os.path.join(config["root_path"], config["train_data_path"]))

    X = data.drop(columns='Com_Gold')
    y = data['Com_Gold']

    test_size = config["pred_len"]

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    # 클래스 사용
    lasso = lasso_model(alpha=0.1)
    lasso.fit(X_train, y_train)

    # 선택된 feature 확인
    selected_features = lasso.get_selected_features()
    print("Selected features:", selected_features)
    print("len of selected features:", len(selected_features))

    # 계수 확인
    coefficients = lasso.get_coefficients()
    print(coefficients[coefficients != 0])

### 최종 실험 함수
def run_model(config):
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

            # ✅ wandb 기록 조건 추가
            if config.get("use_wandb", False):
                wandb.init(
                    project=config.get("wandb_project", "default_project"),
                    name=f"{model_name}_{cd_name}",
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

            tuner = HyperParameterTuner(config, param_ranges, n_splits=2, n_trials=5)
            tuner.run_study()
            tuner.train_final_model()

            if config.get("use_wandb", False):
                wandb.finish()


if __name__ == '__main__':
    # run_causal_discovery(base_config)
    # base_config["wandb_project"] = "90d_test"
    run_model(base_config)