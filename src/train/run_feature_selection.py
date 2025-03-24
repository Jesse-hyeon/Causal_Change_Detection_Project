import pandas as pd
import json
import os

from src.causal_discovery.Lasso import lasso_model

### config(JSON 파일) 불러오기
config_path = "/Users/choeseoheon/Desktop/Causal-Discovery/src/Arg/config.json"
with open(config_path, "r") as f:
    base_config = json.load(f)

### 데이터 관련 정보 저장
raw_data = pd.read_csv(os.path.join(base_config["root_path"], base_config["data_path"]))
print(raw_data.columns)

if __name__ == '__main__':
    # base_config["wandb_project"] = "test1"
    # run(base_config)
    # 데이터 준비
    data = raw_data.iloc[:-90, :]

    X = data.drop(columns='Com_Gold')
    y = data['Com_Gold']

    test_size = 90
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    # 클래스 사용
    lasso_model = lasso_model(alpha=0.1)
    lasso_model.fit(X_train, y_train)

    # 평가
    rmse = lasso_model.evaluate(X_test, y_test)
    print("RMSE:", rmse)

    # 선택된 feature 확인
    selected_features = lasso_model.get_selected_features()
    print("Selected features:", selected_features)
    print("len of selected features:", len(selected_features))

    # 계수 확인
    coefficients = lasso_model.get_coefficients()
    print(coefficients[coefficients != 0])