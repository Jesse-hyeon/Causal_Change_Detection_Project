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
    file_path = os.path.join(base_path, "input", "ALL_data_add.csv")

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
        # data = data[-600: ]  # 일별 데이터일 경우 최근 600일만 사용
        data = data[data.index >= "2020-04-06"]
        # data = data

    # 스케일링
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    return data

# JSON 저장 (줄바꿈 없이 깔끔하게)
def save_pretty_json(data, path):
    with open(path, 'w') as f:
        f.write('{\n')
        last_key = list(data.keys())[-1]
        for key, value in data.items():
            f.write(f'    "{key}": [\n')
            for i, item in enumerate(value):
                if isinstance(item, (list, tuple)):
                    line = f'        ["{item[0]}", {item[1]}]'
                else:
                    line = f'        "{item}"'
                if i != len(value) - 1:
                    line += ','
                f.write(line + '\n')
            if key != last_key:
                f.write('    ],\n')
            else:
                f.write('    ]\n')
        f.write('}\n')

# === 실행 부분 ===
if __name__ == "__main__":
    print("Test script started!")

    methods = ["Lasso", "PCMCI", "CBNB"]
    feature_sets = {}

    data = load_data(base_config, method=methods[0])
    all_features = ["date"] + list(data.columns)
    if "Com_Gold" in all_features:
        all_features.remove("Com_Gold")
    all_features.append("Com_Gold")
    feature_sets["all"] = all_features

    target = "Com_Gold"
    for method in methods:
        print(f"\n=== Testing Feature Selection Method: {method} ===")

        # ✅ 각 방법(method)마다 fresh하게 데이터 다시 로드!
        data = load_data(base_config, method=method)

        selector = FeatureSelector(data=data, target_col=target, method=method)

        try:
            result = selector.select_features()
            com_gold_causes = result["com_gold_causes"]

            if isinstance(com_gold_causes[0], tuple):
                processed = [("date", 0)] + [feat for feat in com_gold_causes if feat[0] != "Com_Gold"]
                processed.append(("Com_Gold", 0))
            else:
                processed = ["date"] + [feat for feat in com_gold_causes if feat != "Com_Gold"]
                processed.append("Com_Gold")

            feature_sets[method] = processed

        except Exception as e:
            print(f"Error occurred while testing method {method}: {e}")

    # === 최종 저장 ===
    print("\nFinal feature sets:")
    for key, value in feature_sets.items():
        print(f"{key}: {value}")

    output_dir = os.path.join(base_path, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "feature_sets.json")

    save_pretty_json(feature_sets, output_path)

    print(f"\nFeature sets JSON file saved at: {output_path}")