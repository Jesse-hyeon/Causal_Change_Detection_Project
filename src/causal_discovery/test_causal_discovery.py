import os
import pandas as pd
from src.train.run_feature_selection import FeatureSelector
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
def load_data():
    file_path = "/Users/choeseoheon/Desktop/Causal-Discovery/input/train_len.csv"
    data = pd.read_csv(file_path)
    data.dropna(axis=0, how='any', inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')

    if 'date' in data.columns:
        data.drop(columns=['date'], inplace=True)

    # 스케일링
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data

# 실행 시
if __name__ == "__main__":
    print("Test script started!")

    data = load_data()
    print("데이터 로드 완료. 데이터 shape:", data.shape)
    print("전체 데이터 컬럼 목록:", data.columns.tolist())

    # # 선택하고 싶은 feature 목록 선택
    # selected_features = ['USD_CNY', 'USD_AUD', 'USD_DXY', 'Stocks_US500', 'Stocks_USVIX', 'Stocks_CH50', 'Stocks_SHANGHAI', 'Bonds_CHN_30Y', 'Bonds_CHN_20Y',
    #                      'Bonds_CHN_10Y', 'Bonds_CHN_5Y', 'Bonds_CHN_2Y', 'Bonds_CHN_1Y', 'Bonds_US_10Y', 'Bonds_US_2Y', 'Bonds_US_1Y', 'Bonds_US_3M', 'Bonds_AUS_10Y',
    #                      'Bonds_AUS_1Y', 'Com_CrudeOil', 'Com_BrentCrudeOil', 'Com_Gasoline', 'Com_NaturalGas', 'Com_Silver', 'EPU_GEPU_current', 'EPU_GEPU_ppp',
    #                      'EPU_Australia', 'EPU_Brazil', 'EPU_Canada', 'EPU_Chile', 'EPU_Hybrid_China', 'EPU_France', 'EPU_Germany', 'EPU_UK', 'EPU_US', 'EPU_Mainland_China',
    #                      'Com_Gold']
    target = "Com_Gold"

    # # 타겟변수도 항상 selected_features에 추가하기
    # if target not in selected_features:
    #     selected_features.append(target)
    #
    # # 선택된 feature들만 데이터에서 추출
    # data = data[selected_features]
    # print("선택된 데이터 컬럼 목록:", data.columns.tolist())

    # 사용할 기법 선택
    methods = ["NBCB"]

    for method in methods:
        print(f"\n=== Testing Feature Selection Method: {method} ===")
        selector = FeatureSelector(data=data, target_col=target, method=method)
        try:
            result = selector.select_features()

            if method == "NBCB":
                full_result = result["full_result"]
                com_gold_causes = result["com_gold_causes"]
                print("Com_Gold에 영향을 미치는 변수:", com_gold_causes)

        except Exception as e:
            print(f"Error occurred while testing method {method}: {e}")