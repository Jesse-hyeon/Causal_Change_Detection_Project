import sys
import os
sys.path.append(os.path.abspath("src"))  # src 폴더를 경로에 추가

import json
import random
import pandas as pd
import numpy as np
import wandb

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

### 인과추론 코드 완성되기 전에 임의로 만드는 causal_discovery_list
feature_sets = {
    "Lasso": [
        "date", "USD_CNY", "USD_AUD", "USD_DXY", "Stocks_US500", "Stocks_USVIX", "Stocks_CH50", "Stocks_SHANGHAI",
        "Bonds_CHN_30Y", "Bonds_CHN_20Y", "Bonds_CHN_10Y", "Bonds_CHN_5Y", "Bonds_CHN_2Y", "Bonds_CHN_1Y",
        "Bonds_US_10Y", "Bonds_US_2Y", "Bonds_US_1Y", "Bonds_US_3M", "Bonds_AUS_10Y", "Bonds_AUS_1Y",
        "Com_CrudeOil", "Com_BrentCrudeOil", "Com_Gasoline", "Com_NaturalGas", "Com_Silver", "EPU_GEPU_current",
        "EPU_GEPU_ppp", "EPU_Australia", "EPU_Brazil", "EPU_Canada", "EPU_Chile", "EPU_Hybrid_China", "EPU_France",
        "EPU_Germany", "EPU_UK", "EPU_US", "EPU_Mainland_China", "Com_Gold"
    ],
    "PCMCI": [
        "date", "USD_KRW", "USD_JPY", "USD_DXY", "Stocks_US500", "Stocks_CH50", "Stocks_CSI300", "Stocks_HK50",
        "Bonds_CHN_10Y", "Bonds_CHN_5Y", "Bonds_US_10Y", "Bonds_US_2Y", "Com_CrudeOil", "Com_NaturalGas",
        "Com_IronOre", "Com_Copper", "EPU_GEPU_current", "EPU_France", "EPU_Germany", "EPU_Japan", "Com_Gold"
    ],
    "NBCB": [
        "date", "Bonds_CHN_10Y", "Bonds_CHN_1Y", "Bonds_CHN_2Y", "Bonds_CHN_5Y", "Bonds_US_10Y", "Bonds_US_1Y",
        "Bonds_US_2Y", "Bonds_US_3M", "Com_BrentCrudeOil", "Com_Gasoline", "Com_Silver", "EPU_GEPU_ppp",
        "Stocks_CH50", "Stocks_SHANGHAI", "Stocks_US500", "Stocks_USVIX", "USD_AUD", "USD_CNY", "USD_DXY", "Com_Gold"
    ]
}

### 인과 발견, 모델 리스트
causal_discovery_list = ["Lasso", "PCMCI", "NBCB"]
model_list = ["lstm", "transformer", "ns_transformer"]

### 최종 실험 함수
### 최종 실험 함수
def run(config):
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

            # ✅ wandb.init()에서 실험 이름 설정
            wandb.init(
                project=config.get("wandb_project", "default_project"),
                name=f"{model_name}_{cd_name}",  # wandb에 모델_인과발견방법론으로 저장
                config=config
            )

            if config["model"] in config["former_model"]:  # Transformer 기반 모델인 경우
                len_in_former = len(config["feature_set"]) - 1
                config["enc_in"] = len_in_former
                config["dec_in"] = len_in_former
                tuner = HyperParameterTuner(config, param_ranges, n_splits=1, n_trials=1)
                tuner.run_study()
                tuner.train_final_model()
            else:  # RNN 기반 모델인 경우
                len_in_rnn = len(config["feature_set"]) - 1 + 3  # 미래공변량 피처 더하기
                config["enc_in"] = len_in_rnn
                config["dec_in"] = len_in_rnn
                tuner = HyperParameterTuner(config, param_ranges, n_splits=1, n_trials=1)
                tuner.run_study()
                tuner.train_final_model()

            # ✅ 실험 종료 후 wandb.finish() 호출
            wandb.finish()


if __name__ == '__main__':
    base_config["wandb_project"] = "test"
    run(base_config)