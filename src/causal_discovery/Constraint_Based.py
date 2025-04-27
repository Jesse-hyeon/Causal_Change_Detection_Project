import pandas as pd
import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from statsmodels.stats.outliers_influence import variance_inflation_factor


class pcmci_model:
    def __init__(self, data: pd.DataFrame, tau_max=3, alpha=0.05, target_var="Com_Gold"):
        self.data = data
        self.tau_max = tau_max
        self.alpha = alpha
        self.target_var = target_var

    def select_features_pcmci(self):
        # 1. 데이터 변환
        dataframe = pp.DataFrame(self.data.values, var_names=list(self.data.columns))

        # 2. 독립성 검정 함수
        ind_test = ParCorr(significance='analytic')

        # 3. PCMCI 구성
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=ind_test
        )

        # 4. run_pcmci 실행
        results = pcmci.run_pcmci(
            tau_max=self.tau_max,
            pc_alpha=self.alpha
        )

        # 5. 유의미한 인과간선 출력
        pcmci.print_significant_links(
            p_matrix=results['p_matrix'],
            val_matrix=results['val_matrix'],
            alpha_level=self.alpha
        )

        # 6. target 변수에 대한 feature selection
        var_names = list(self.data.columns)
        target_idx = var_names.index(self.target_var)

        p_matrix = results['p_matrix']
        causal_features = []

        for col_idx, col_name in enumerate(var_names):
            if col_name == self.target_var:
                continue
            for tau in range(self.tau_max + 1):
                pval = p_matrix[target_idx, col_idx, tau]
                if pval < self.alpha:
                    causal_features.append((col_name, -tau))

        return causal_features

    def filter_features_by_multicollinearity(self, causal_features, method="mean", vif_thresh=5.0, corr_thresh=0.8):
        """
        select_features_pcmci() 결과를 받아 VIF + 상관관계 기반으로 다중공선성 제거.
        method: 'mean' 또는 'sum' (인과계수 계산 방식 선택)
        """
        var_names = list(self.data.columns)
        target_idx = var_names.index(self.target_var)

        dataframe = pp.DataFrame(self.data.values, var_names=var_names)
        ind_test = ParCorr(significance='analytic')
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ind_test)
        results = pcmci.run_pcmci(tau_max=self.tau_max, pc_alpha=self.alpha)

        p_matrix = results["p_matrix"]
        val_matrix = results["val_matrix"]

        selected_vars = sorted(set([var for var, _ in causal_features]))

        # 인과계수 모으기
        coef_map = {}
        for col_name in selected_vars:
            col_idx = var_names.index(col_name)
            coefs = []
            for tau in range(self.tau_max + 1):
                pval = p_matrix[target_idx, col_idx, tau]
                if pval < self.alpha:
                    coef = abs(val_matrix[target_idx, col_idx, tau])
                    coefs.append(coef)
            if coefs:
                if method == "sum":
                    coef_map[col_name] = sum(coefs)
                else:  # default: mean
                    coef_map[col_name] = np.mean(coefs)

        # VIF 계산 대상 DataFrame
        X = self.data[selected_vars]
        vif_vals = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_df = pd.DataFrame({"var": X.columns, "vif": vif_vals})
        high_vif = vif_df[vif_df["vif"] > vif_thresh]["var"].tolist()

        # 상관관계 그룹핑
        corr_matrix = X[high_vif].corr().abs()
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

        # 그룹 내 인과계수 기준 가장 중요한 변수 선택
        selected = []
        for group in groups:
            best = max(group, key=lambda var: coef_map.get(var, 0))
            selected.append(best)

        # 그룹에 포함되지 않은 나머지 변수 추가
        ungrouped = set(selected_vars) - set().union(*groups)
        selected += list(ungrouped)

        return selected