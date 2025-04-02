import numpy as np
import pandas as pd

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from lingam import VARLiNGAM

class nbcb_model:
    def __init__(self,
                 data: pd.DataFrame,
                 tau_max: int = 3,
                 sig_level: float = 0.05,
                 threshold: float = 0.01,
                 linear: bool = True):

        self.data = data
        self.tau_max = tau_max
        self.sig_level = sig_level
        self.threshold = threshold
        self.linear = linear

        self.target_var = "Com_Gold"
        self.window_causal_graph_dict = {self.target_var: []}

    def _noise_based(self):
        print("=== [NBCBw] Noise-Based Step: VarLiNGAM ===")
        var_names = list(self.data.columns)
        model = VARLiNGAM(lags=self.tau_max, criterion="HQIC", prune=False)
        model.fit(self.data.values)

        adjacency_mats = model.adjacency_matrices_
        n_vars = len(var_names)
        target_idx = var_names.index(self.target_var)

        for lag_idx, mat in enumerate(adjacency_mats, start=1):
            for j in range(n_vars):
                if j == target_idx:
                    continue
                coef = mat[target_idx, j]
                if abs(coef) > self.threshold:
                    cause_name = var_names[j]
                    self.window_causal_graph_dict[self.target_var].append((cause_name, -lag_idx))

    def _constraint_based(self):
        print("=== [NBCBw] Constraint-Based Step: PCMCI ===")

        nb_candidates = set(self.window_causal_graph_dict[self.target_var])
        var_names = list(self.data.columns)

        # 대상 변수 + 후보 원인만 필터링
        used_vars = [self.target_var] + [v for (v, _) in nb_candidates if v in var_names]
        filtered_data = self.data[used_vars]

        print("=== [DEBUG] 필터링된 변수 목록 ===")
        print(used_vars)

        dataframe = pp.DataFrame(
            data=filtered_data.values,
            var_names=used_vars
        )
        cond_ind_test = ParCorr(significance='analytic')
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=1
        )

        results = pcmci.run_pcmci(
            tau_min=0,
            tau_max=self.tau_max,
            pc_alpha=self.sig_level
        )

        p_matrix = results["p_matrix"]
        target_idx = used_vars.index(self.target_var)

        confirmed_causal_graph_dict = {self.target_var: []}

        for (cause_name, lag_val) in nb_candidates:
            if cause_name not in used_vars:
                continue
            j = used_vars.index(cause_name)
            tau = -lag_val

            # 에러 방지
            if tau >= p_matrix.shape[2]:
                print(f"[⚠ 경고] tau={tau}가 p_matrix에 없음. 무시됨.")
                continue

            pval = p_matrix[target_idx, j, tau]

            if pval < self.sig_level:
                confirmed_causal_graph_dict[self.target_var].append((cause_name, lag_val))
                print(f"[✓ CB 통과] {cause_name} (lag {lag_val}) → {self.target_var} | p={pval:.5f}")
            else:
                print(f"[✗ CB 탈락] {cause_name} (lag {lag_val}) → {self.target_var} | p={pval:.5f}")

        self.window_causal_graph_dict = confirmed_causal_graph_dict

        print("\n=== [CB] Final window_causal_graph_dict (filtered) ===")
        print(f"{self.target_var}: {self.window_causal_graph_dict[self.target_var]}")

    def run(self):
        self._noise_based()
        self._constraint_based()

        com_gold_causes = [cause for (cause, lag) in self.window_causal_graph_dict[self.target_var]]
        com_gold_causes = list(dict.fromkeys(com_gold_causes))

        return self.window_causal_graph_dict, com_gold_causes

class cbnb_model:
    def __init__(self,
                 data: pd.DataFrame,
                 tau_max: int = 3,
                 sig_level: float = 0.1,
                 threshold: float = 0.01,
                 linear: bool = True):

        self.data = data
        self.tau_max = tau_max
        self.sig_level = sig_level
        self.threshold = threshold
        self.linear = linear

        self.target_var = "Com_Gold"
        self.window_causal_graph_dict = {self.target_var: []}

    def _constraint_based(self):
        print("=== [CBNB] Constraint-Based Step: PCMCI ===")
        var_names = list(self.data.columns)
        dataframe = pp.DataFrame(data=self.data.values, var_names=var_names)
        cond_ind_test = ParCorr(significance='analytic')

        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=1
        )

        results = pcmci.run_pcmci(
            tau_min=0,
            tau_max=self.tau_max,
            pc_alpha=self.sig_level
        )

        p_matrix = results["p_matrix"]
        target_idx = var_names.index(self.target_var)

        cb_candidates = []

        for j in range(len(var_names)):
            if j == target_idx:
                continue
            for tau in range(self.tau_max + 1):
                pval = p_matrix[j, target_idx, tau]

                if pval < self.sig_level:
                    cause_name = var_names[j]
                    lag_val = -tau
                    cb_candidates.append((cause_name, lag_val))
                    print(f"[✓ CB 후보] {cause_name} (lag {lag_val}) → {self.target_var} | p={pval:.5f}")

        self.window_causal_graph_dict[self.target_var] = cb_candidates

    def _noise_based(self):
        print("\n=== [CBNB] Noise-Based Step: VarLiNGAM (후보 검증) ===")
        var_names = list(self.data.columns)

        # 🔧 수정된 부분: criterion=None으로 설정
        model = VARLiNGAM(lags=self.tau_max, criterion="HQIC", prune=False)
        model.fit(self.data.values)
        adjacency_mats = model.adjacency_matrices_

        actual_tau_max = len(adjacency_mats)
        print(f"[DEBUG] VarLiNGAM 추정된 시차 수 (강제 사용): {actual_tau_max}")

        target_idx = var_names.index(self.target_var)
        cb_candidates = self.window_causal_graph_dict[self.target_var]
        final_result = []

        for (cause_name, lag_val) in cb_candidates:
            cause_idx = var_names.index(cause_name)
            tau = -lag_val

            # 유효성 검사
            if tau <= 0 or tau > actual_tau_max:
                print(f"[건너뜀] Invalid tau={tau} for {cause_name} (lag {lag_val}) | 설정된 tau_max={self.tau_max}, 실제={actual_tau_max}")
                continue

            coef = adjacency_mats[tau - 1][target_idx, cause_idx]
            print(f"[계수 확인] {cause_name} (lag {lag_val}) → {self.target_var} | coef={coef:.5f}")

            if abs(coef) > self.threshold:
                final_result.append((cause_name, lag_val))
                print(f"[✓ NB 통과] {cause_name} (lag {lag_val}) → {self.target_var} | coef={coef:.5f}")
            else:
                print(f"[✗ NB 탈락] {cause_name} (lag {lag_val}) → {self.target_var} | coef={coef:.5f}")

        self.window_causal_graph_dict[self.target_var] = final_result

        print("\n=== [CBNB] 최종 window_causal_graph_dict ===")
        print(f"{self.target_var}: {self.window_causal_graph_dict[self.target_var]}")

    def run(self):
        self._constraint_based()
        self._noise_based()

        com_gold_causes = [cause for (cause, lag) in self.window_causal_graph_dict[self.target_var]]
        com_gold_causes = list(dict.fromkeys(com_gold_causes))

        return self.window_causal_graph_dict, com_gold_causes
