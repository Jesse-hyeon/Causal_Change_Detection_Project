import platform
import os
import json

import pandas as pd
from src.train.feature_select import FeatureSelector
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller


### config(JSON 파일) 불러오기
if platform.system() == 'Windows':
    base_path = 'C:/Users/T-Lab_Public/PycharmProjects/Causal-Discovery'
else:
    base_path = '/Users/choeseoheon/Desktop/Causal-Discovery'

config_path = os.path.join(base_path, "src/Arg/config.json")

with open(config_path, "r") as f:
    base_config = json.load(f)

# 데이터 불러오기
def load_data(config, method=None):
    file_path = os.path.join(base_path, "input", "ALL_data.csv")

    raw_data = pd.read_csv(file_path, parse_dates=["date"])
    raw_data.set_index("date", inplace=True)

    raw_data.dropna(axis=0, how='any', inplace=True)

    if method == "Lasso":
        data = raw_data
    else:
        test_size = config["CD_pred_len"]
        data = raw_data[:-test_size]

    if config["CD_freq"] == "w":
        data = data.resample("W").mean()
    elif config["CD_freq"] == "m":
        data = data.resample("ME").mean()
    else:
        # data = data[-1000: ]  # 일별 데이터일 경우 최근 600일만 사용
        data = data[data.index >= "2020-01-01"]
        # data = data

    # 스케일링
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    return data

# 실행 시
if __name__ == "__main__":
    print("Test script started!")

    # 사용할 기법 선택
    # "Lasso", "PCMCI", "VARLiNGAM", "NBCB", "CBNB"
    methods = ["VARLiNGAM"]

    # 결과 feature들을 저장할 딕셔너리
    feature_sets = {}

    # 모든 변수(feature)를 담은 "all" key 생성
    data = load_data(base_config, method=methods[0])
    all_features = ["date"] + list(data.columns)
    if "Com_Gold" in all_features:
        all_features.remove("Com_Gold")
    all_features.append("Com_Gold")
    feature_sets["all"] = all_features

    # 각 기법(method)에 따른 feature selection 결과 처리
    target = "Com_Gold"
    for method in methods:
        print("data shape: ", data.shape)
        print(f"\n=== Testing Feature Selection Method: {method} ===")

        selector = FeatureSelector(data=data, target_col=target, method=method)
        try:
            result = selector.select_features()

            com_gold_causes = result["com_gold_causes"]
            print(f"[{method}] Com_Gold에 영향을 미치는 변수:", com_gold_causes)
            # feature 정리
            com_gold_causes = [feat for feat in com_gold_causes if feat != "Com_Gold"]

            # "date"가 없으면 맨 앞에 추가 (정상적인 리스트 연결!)
            if "date" not in com_gold_causes:
                com_gold_causes = ["date"] + com_gold_causes

            # "Com_Gold"는 무조건 맨 뒤에 추가
            com_gold_causes.append("Com_Gold")

            feature_sets[method] = com_gold_causes
            print(f"Selected features for {method}: {com_gold_causes}")

        except Exception as e:
            print(f"Error occurred while testing method {method}: {e}")

    # 최종 feature_sets 출력
    print("\nFinal feature sets:")
    for key, value in feature_sets.items():
        print(f"{key}: {value}")

    # feature_sets를 JSON 파일로 저장 (저장 경로는 base_path 하위 output 폴더)
    output_dir = os.path.join(base_path, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "feature_sets.json")
    with open(output_path, "w") as f:
        json.dump(feature_sets, f, indent=4)
    print(f"\nFeature sets JSON file saved at: {output_path}")
