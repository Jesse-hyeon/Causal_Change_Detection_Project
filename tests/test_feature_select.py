import os
import pandas as pd
from src.train.feature_select import FeatureSelector
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

    # 선택하고 싶은 feature 목록 선택
    selected_features = ['Bonds_AUS_10Y', 'Bonds_AUS_1Y', 'Bonds_CHN_10Y', 'Bonds_CHN_1Y', 'Bonds_CHN_20Y', 'Bonds_CHN_2Y',
    'Bonds_CHN_30Y', 'Bonds_CHN_5Y', 'Bonds_BRZ_10Y', 'Bonds_BRZ_1Y', 'Bonds_IND_10Y', 'Bonds_IND_1Y', 'Bonds_KOR_10Y',
    'Bonds_KOR_1Y', 'Bonds_US_10Y', 'Bonds_US_1Y', 'Bonds_US_2Y', 'Bonds_US_3M', 'Com_Coking_Coal', 'Com_Barley', 'Com_Corn',
    'Com_Cocoa', 'Com_Cheese', 'Com_CrudeOil', 'Com_BrentCrudeOil', 'Com_Cotton', 'Com_Milk', 'Com_Copper', 'Com_HRC_Steel',
    'Com_Steel', 'Com_Lumber', 'Com_Aluminum', 'Com_Nickel', 'Com_NaturalGas', 'Com_Oat', 'Com_Wool', 'Com_PalmOil', 'Com_Rice',
    'Com_Canola', 'Com_Soybeans', 'Com_Sugar', 'Com_Iron_Ore', 'Com_SunflowerOil', 'Com_Uranium', 'Com_Wheat', 'Com_Silver',
    'Com_Coal', 'Com_Gold', 'Com_Gasoline', 'Com_OrangeJuice', 'Com_Coffee', 'EX_AUD_USD', 'EX_USD_BRL', 'EX_USD_CNY', 'EX_INR_USD',
    'EX_USD_JPY', 'EX_USD_KRW', 'Idx_DxyUSD', 'Idx_HangSeng', 'Com_LME_Index', 'Idx_Shanghai', 'Idx_CSI300', 'Idx_SnPGlobal1200',
    'Idx_SnP500', 'Idx_Shanghai50', 'Idx_SnPVIX', 'Idx_CH50', 'EPU_GEPU_current', 'EPU_GEPU_ppp', 'EPU_Australia', 'EPU_Brazil',
    'EPU_Canada', 'EPU_Chile', 'EPU_Hybrid China', 'EPU_France', 'EPU_Germany', 'EPU_Greece', 'EPU_India', 'EPU_Ireland', 'EPU_Italy',
    'EPU_Japan', 'EPU_Korea', 'EPU_Pakistan', 'EPU_Russia', 'EPU_Spain', 'EPU_Singapore', 'EPU_UK', 'EPU_US', 'EPU_Mainland China',
    'EX_EUR_USD', 'EX_CAD_JPY', 'Com_Palladium', 'Com_Platinum', 'Idx_DowJones', 'Idx_NASDAQ', 'Idx_MOVE', 'Idx_CBOE_VIX', 'Idx_US_PMI',
    'Idx_US_IPI', 'Idx_US_IPI_chg', 'Idx_US_CPI', 'Idx_US_CPI_chg', 'Idx_US_CCI', 'Idx_US_GDP_Deflator', 'Idx_FEDFUNDS', 'Idx_US_UnemploymentRate_chg']
    target = "Com_Gold"

    # # 타겟변수도 항상 selected_features에 추가하기
    if target not in selected_features:
        selected_features.append(target)

    # 선택된 feature들만 데이터에서 추출
    data = data[selected_features]
    print("선택된 데이터 컬럼 목록:", data.columns.tolist())

    # 사용할 기법 선택
    # Lasso, PCMCIPlUS, VARLiNGAM, NBCB
    methods = ["VARLiNGAM"]

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