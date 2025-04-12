import pandas as pd
from lingam import VARLiNGAM
from statsmodels.stats.outliers_influence import variance_inflation_factor

class varlingam_model:
    def __init__(self, data: pd.DataFrame, tau_max=3, target_var="Com_Gold"):
        self.data = data
        self.tau_max = tau_max
        self.target_var = target_var

        self.model = VARLiNGAM(
            lags=self.tau_max,
            criterion="BIC",
            prune=True
        )

    def fit(self):
        self.model.fit(self.data.values)
        self.var_names = list(self.data.columns)
        self.adjacency_mats = self.model.adjacency_matrices_
        return self

    def get_adjacency_df(self):
        if not hasattr(self, 'adjacency_mats'):
            raise ValueError("Model is not fitted yet. Call `fit()` first.")

        rows = []
        for lag_idx, mat in enumerate(self.adjacency_mats):
            time_lag = 0 if lag_idx == 0 else -lag_idx
            for i in range(len(self.var_names)):
                for j in range(len(self.var_names)):
                    rows.append({
                        "lag": time_lag,
                        "from": self.var_names[j],
                        "to": self.var_names[i],
                        "coef": mat[i, j]
                    })
        return pd.DataFrame(rows, columns=["lag", "from", "to", "coef"])

    def select_features(self, return_only_var_names=False):
        if not hasattr(self, 'adjacency_mats'):
            raise ValueError("Model is not fitted yet. Call `fit()` first.")

        feature_set = set()
        target_idx = self.var_names.index(self.target_var)
        for lag_idx, mat in enumerate(self.adjacency_mats):
            for cause_idx in range(len(self.var_names)):
                if mat[target_idx, cause_idx] != 0 and cause_idx != target_idx:
                    time_lag = 0 if lag_idx == 0 else -lag_idx
                    feature_set.add((self.var_names[cause_idx], time_lag))

        final_features = sorted(list(feature_set))
        if return_only_var_names:
            return sorted({var for var, lag in final_features})
        else:
            return final_features

    def filter_selected_features(self, selected_vars, vif_thresh=5, corr_thresh=0.8):
        """
        select_features(return_only_var_names=True) 결과를 받아,
        그 안에서만 다중공선성(VIF) + 상관관계 필터링을 적용하여
        최종 선택된 변수 리스트 반환.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        df = self.data[selected_vars].copy()

        # Step 1: VIF 계산
        vif_vals = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        vif_df = pd.DataFrame({"var": df.columns, "vif": vif_vals})
        high_vif = vif_df[vif_df["vif"] > vif_thresh]["var"].tolist()

        # Step 2: 상관관계 그룹핑
        corr_matrix = df[high_vif].corr().abs()
        groups = []
        visited = set()

        for col in corr_matrix.columns:
            if col in visited:
                continue
            group = set([col])
            for other in corr_matrix.columns:
                if col != other and corr_matrix.loc[col, other] > corr_thresh:
                    group.add(other)
                    visited.add(other)
            visited.update(group)
            if len(group) > 1:
                groups.append(group)

        # Step 3: 각 그룹 내에서 인과계수 가장 큰 변수 선택
        coef_map = {}
        for lag_idx, mat in enumerate(self.adjacency_mats):
            for cause_idx, cause_var in enumerate(self.var_names):
                if cause_var in selected_vars and cause_var != self.target_var:
                    coef = mat[self.var_names.index(self.target_var), cause_idx]
                    if coef != 0:
                        coef_map[cause_var] = coef_map.get(cause_var, 0) + abs(coef)

        selected = []
        for group in groups:
            best = max(group, key=lambda var: abs(coef_map.get(var, 0)))
            selected.append(best)

        # Step 4: 그룹에 속하지 않은 나머지 변수 추가
        ungrouped = set(selected_vars) - set().union(*groups)
        selected += list(ungrouped)

        return selected

