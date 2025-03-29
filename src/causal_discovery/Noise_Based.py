import pandas as pd
from lingam import VARLiNGAM

class varlingam_model:
    def __init__(self, data: pd.DataFrame, tau_max=3, threshold=0.1, target_var="Com_Gold"):
        self.data = data
        self.tau_max = tau_max
        self.threshold = threshold
        self.target_var = target_var

        # 모델 생성
        self.model = VARLiNGAM(
            lags=self.tau_max,
            criterion='bic',
            prune=False
        )

    # 모델 훈련
    def fit(self):
        self.model.fit(self.data.values)
        self.var_names = list(self.data.columns)  # 변수명 리스트
        self.adjacency_mats = self.model.adjacency_matrices_
        return self

    # 전체 adjacency matrix를 pandas DataFrame 형태로 반환 (이 부분은 조금 더 봐야할듯)
    def get_adjacency_df(self):
        if not hasattr(self, 'adjacency_mats'):
            raise ValueError("Model is not fitted yet. Call `fit()` first.")

        rows = []
        for lag_idx, mat in enumerate(self.adjacency_mats, start=1):
            # mat.shape == (n_vars, n_vars), mat[i,j]: lag_idx 시점에서 j->i
            for i in range(len(self.var_names)):
                for j in range(len(self.var_names)):
                    effect = mat[i, j]
                    rows.append({
                        "lag": lag_idx,
                        "from": self.var_names[j],
                        "to": self.var_names[i],
                        "coef": effect
                    })
        df = pd.DataFrame(rows, columns=["lag", "from", "to", "coef"])
        return df


    def select_features(self, return_only_var_names=False):
        """
        target_var가 None이면, 모든 edge를 반환.
        target_var가 주어지면, target_var(t)에 직접 영향이 있는 (원인) 변수만 추출.
        """
        if not hasattr(self, 'adjacency_mats'):
            raise ValueError("Model is not fitted yet. Call `fit()` first.")

        feature_set = set()
        var_names = self.var_names

        if self.target_var is None:
            # 모든 인과관계 반환
            return self.get_adjacency_df()

        target_idx = var_names.index(self.target_var)
        # adjacency_mats[l][i, j] => j -> i, lag = l+1
        for lag_idx, mat in enumerate(self.adjacency_mats, start=1):
            for cause_idx in range(len(var_names)):
                coef = mat[target_idx, cause_idx]
                if abs(coef) > self.threshold and (cause_idx != target_idx):
                    feature_set.add((var_names[cause_idx], -lag_idx))

        final_features = sorted(list(feature_set))

        if return_only_var_names:
            feature_names = sorted({var for var, lag in final_features})
            return feature_names # 이름만 출력
        else:
            return final_features # (var_name, -lag)
