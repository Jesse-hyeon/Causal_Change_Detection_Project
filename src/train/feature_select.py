import platform
import json

import pandas as pd
import numpy as np
from typing import Dict, Any, List

### config(JSON 파일) 불러오기
if platform.system() == 'Windows':
    base_path = 'C:/Users/T-Lab_Public/PycharmProjects/Causal-Discovery'
else:
    base_path = '/Users/choeseoheon/Desktop/Causal-Discovery'

config_path = base_path + "/src/Arg/config.json"

with open(config_path, "r") as f:
    base_config = json.load(f)

class FeatureSelector:
    def __init__(self, data, target_col, method="CFS"):
        self.data = data
        self.target_col = target_col
        self.method = method

    def _select_features_lasso(self, config, alpha="auto"):
        from src.causal_discovery.Traditional import lasso_model
        target = config["target"]
        X = self.data.drop(columns=target)
        y = self.data[target]

        test_size = config["pred_len"]
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]

        lasso = lasso_model(config=config, alpha=alpha)
        lasso.fit(X_train, y_train)

        # 다중공선성 제거 X
        # selected_features = list(lasso.get_selected_features())

        # 다중공선성 제거 O
        selected_features = lasso.filter_features_by_multicollinearity(X_train)

        print(f"Lasso selected features: {selected_features}")
        return {"com_gold_causes": selected_features}

    def _select_features_pcmci(self, config, threshold=0.05):
        from src.causal_discovery.Constraint_Based import pcmci_model

        selector = pcmci_model(
            data=self.data,
            tau_max=config['tau_max'],
            alpha=threshold,
            target_var=self.target_col
        )

        # 1. ("feature", lag) 튜플 리스트 확보
        causal_features = selector.select_features_pcmci()

        # 2. 다중공선성 제거
        # feature_name만 따로 뽑아서 다중공선성 필터링
        feature_names = selector.filter_features_by_multicollinearity(causal_features)

        # 3. 다중공선성 통과한 feature만 남기기
        # ("feature", lag) 튜플 중에서 feature_name이 살아남은 것만 유지
        selected_causal_features = [(var, lag) for (var, lag) in causal_features if var in feature_names]

        print(f"PCMCI selected features for {self.target_col}: {selected_causal_features}")

        return {
            "com_gold_causes": selected_causal_features  # <-- 튜플 형태로 반환!
        }

    def _select_features_varlingam(self, config):
        from src.causal_discovery.Noise_Based import varlingam_model

        selector = varlingam_model(
            data=self.data,
            tau_max=config['tau_max'],
            target_var=self.target_col
        )

        selector.fit()

        # 1. ("feature", lag) 튜플 리스트 확보
        causal_features = selector.select_features(return_only_var_names=False)  # <-- 튜플 형태로 받아오기!

        # 2. 다중공선성 제거
        feature_names = list(selector.filter_selected_features(
            [var for (var, lag) in causal_features]
        ))

        # 3. 다중공선성 통과한 feature만 남기기
        selected_causal_features = [(var, lag) for (var, lag) in causal_features if var in feature_names]

        print(f"[VarLiNGAM] selected features for {self.target_col}: {selected_causal_features}")

        return {
            "com_gold_causes": selected_causal_features  # <-- 튜플 형태로 반환!
        }


    def _select_features_cbnb(self, config):
        from src.causal_discovery.Hybrid import CBNBe

        model = CBNBe(
            data=self.data,
            tau_max=config['tau_max'],
            sig_level=0.05,
            linear=True
        )

        model.run()

        # 1. ("feature", lag) 튜플 리스트 확보
        causal_features = model.window_causal_graph_dict[self.target_col]

        # 2. 다중공선성 제거
        # -> lag 무시하고 feature 이름만 보고 제거할지, 아니면 lag별로 볼지 결정해야 함
        #    여기서는 기존처럼 feature 이름 단위로 제거한다고 가정
        feature_names = CBNBe.filter_cbnb_by_multicollinearity(self.data, causal_features)

        # 3. 다중공선성 통과한 feature만 남기기
        # ("feature", lag) 리스트에서 feature_name 필터링
        selected_causal_features = [(var, lag) for (var, lag) in causal_features if var in feature_names]

        print(f"CBNB selected features for {self.target_col}: {selected_causal_features}")

        return {
            "full_result": model,
            "com_gold_causes": selected_causal_features  # <-- 튜플 형태로 반환!
        }

    def select_features(self):
        if self.method == "Lasso":
            return self._select_features_lasso(config = base_config)
        elif self.method == "PCMCI":
            return self._select_features_pcmci(config = base_config)
        elif self.method == "VARLiNGAM":
            return self._select_features_varlingam(config = base_config)
        elif self.method == "CBNB":
            return self._select_features_cbnb(config = base_config)
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")
