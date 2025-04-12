import pandas as pd
from lingam import VARLiNGAM

class varlingam_model:
    def __init__(self, data: pd.DataFrame, tau_max=3, target_var="Com_Gold"):
        self.data = data
        self.tau_max = tau_max
        self.target_var = target_var

        self.model = VARLiNGAM(
            lags=self.tau_max,
            criterion="BIC", # BIC에 따른 처리
            prune=True # Hybrid에서는 False로 두긴 함,,
        )

    # 모델 훈련
    def fit(self):
        self.model.fit(self.data.values)
        self.var_names = list(self.data.columns)  # 변수명 리스트
        self.adjacency_mats = self.model.adjacency_matrices_
        return self

    def get_adjacency_df(self):
        if not hasattr(self, 'adjacency_mats'):
            raise ValueError("Model is not fitted yet. Call `fit()` first.")

        rows = []
        for lag_idx, mat in enumerate(self.adjacency_mats):
            if lag_idx == 0:
                time_lag = 0
            else:
                time_lag = -lag_idx
            for i in range(len(self.var_names)):
                for j in range(len(self.var_names)):
                    effect = mat[i, j]
                    rows.append({
                        "lag": time_lag,
                        "from": self.var_names[j],
                        "to": self.var_names[i],
                        "coef": effect
                    })
        df = pd.DataFrame(rows, columns=["lag", "from", "to", "coef"])
        return df

    def select_features(self, return_only_var_names=False):
        if not hasattr(self, 'adjacency_mats'):
            raise ValueError("Model is not fitted yet. Call `fit()` first.")

        feature_set = set()
        var_names = self.var_names

        if self.target_var is None:
            return self.get_adjacency_df()

        target_idx = var_names.index(self.target_var)
        for lag_idx, mat in enumerate(self.adjacency_mats):
            for cause_idx in range(len(var_names)):
                if mat[target_idx, cause_idx] != 0 and (cause_idx != target_idx):
                    if lag_idx == 0:
                        time_lag = 0
                    else:
                        time_lag = -lag_idx
                    feature_set.add((var_names[cause_idx], time_lag))

        final_features = sorted(list(feature_set))

        if return_only_var_names:
            feature_names = sorted({var for var, lag in final_features})
            return feature_names
        else:
            return final_features
