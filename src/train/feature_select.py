import platform
import json

import pandas as pd
import numpy as np

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

        # 클래스 사용
        lasso = lasso_model(config=config, alpha=alpha)
        lasso.fit(X_train, y_train)

        selected_features = lasso.get_selected_features()
        print("Selected features:", selected_features)

        return selected_features


    def _select_features_pcmciplus(self, config, threshold=0.05):
        from src.causal_discovery.Constraint_Based import pcmciplus_model

        selector = pcmciplus_model(
            data=self.data,
            tau_max=config['tau_max'],
            alpha=threshold
        )

        causal_features = selector.select_features_pcmci_plus()

        # 변수명만 추출해서 정리
        feature_names = sorted(set([var for var, lag in causal_features]))

        print(f"PCMCI+ selected features for {self.target_col}: {feature_names}")
        return feature_names

    def _select_features_varlingam(self, config, threshold=0.01):
        from src.causal_discovery.Noise_Based import varlingam_model

        selector = varlingam_model(
            data=self.data,
            tau_max=config['tau_max'],
            threshold=threshold,
            target_var=self.target_col
        )

        selector.fit()

        final_features = selector.select_features(return_only_var_names=True) # True -> 이름만

        print(f"[VarLiNGAM] selected features for {self.target_col}: {final_features}")
        return final_features

    def _select_features_nbcb(self, config,  threshold=0.01):
        from src.causal_discovery.Hybrid import nbcbw_model

        nbcb = nbcbw_model(
            data=self.data,
            tau_max=config['tau_max'],
            sig_level=0.05,
            threshold = threshold,
            linear=True,
        )
        full_result, com_gold_causes = nbcb.run()

        if full_result is None:
            raise ValueError("Error: NBCBw.run() returned None. Check the function implementation!")

        return {
            "full_result": full_result,
            "com_gold_causes": com_gold_causes
        }

    def select_features(self):
        if self.method == "Lasso":
            return self._select_features_lasso(config = base_config)
        elif self.method == "PCMCIPlUS":
            return self._select_features_pcmciplus(config = base_config)
        elif self.method == "VARLiNGAM":
            return self._select_features_varlingam(config = base_config)
        elif self.method == "NBCB":
            return self._select_features_nbcb(config = base_config)
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")
