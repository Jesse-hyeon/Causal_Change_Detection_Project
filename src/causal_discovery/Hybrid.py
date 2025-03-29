import numpy as np
import pandas as pd

from src.utils.pcmci_with_bk import PCMCI as PCMCIbk
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from lingam import VARLiNGAM

class nbcbw_model:
    def __init__(self,
                 data: pd.DataFrame, # (행 = 시간, 열 = 변수)
                 tau_max: int = 3, # 최대 시차
                 sig_level: float = 0.05, # PCMCI+에서 사용될 유의수준(pc_alpha)
                 threshold: float = 0.1, # 낮을수록 더 많은 feature을 인과가 있다고 판단
                 linear: bool = True): # VarLiNGAM(선형)

        self.data = data
        self.tau_max = tau_max
        self.sig_level = sig_level
        self.threshold = threshold
        self.linear = linear

        # Noise-Based 결과
        self.forbidden_orientation = []         # (effect_idx, cause_idx)를 기록해서 반대방향 X
        self.window_causal_graph_dict = {       # (원인변수명, -lag) 저장
            col: [] for col in self.data.columns
        }

    def _noise_based(self):
        print("=== [NBCBw] Noise-Based Step: VarLiNGAM ===")
        var_names = list(self.data.columns)
        model = VARLiNGAM(lags=self.tau_max, criterion='bic', prune=False)
        model.fit(self.data.values)

        # 시차별 인과계수 행렬
        adjacency_mats = model.adjacency_matrices_
        n_vars = len(var_names)

        # 인과계수 행렬들을 순서대로 하나씩 꺼내며 돌리기
        for lag_idx, mat in enumerate(adjacency_mats, start=1):
            #
            for i in range(n_vars):       # i: 결과
                for j in range(n_vars):   # j: 원인
                    coef = mat[i, j]

                    # threshold 기준으로 의미있는 인과 판별
                    if abs(coef) > self.threshold:
                        effect_name = var_names[i]
                        cause_name  = var_names[j]

                        # forbidden_orientation
                        self.forbidden_orientation.append((i, j))

                        # window_causal_graph_dict
                        self.window_causal_graph_dict[effect_name].append(
                            (cause_name, -lag_idx)
                        )

    def _constraint_based(self):
        print("=== [NBCBw] Constraint-Based Step: PCMCI+ ===")

        # 1. Tigramite용 data
        dataframe = pp.DataFrame(
            data=self.data.values,
            var_names=list(self.data.columns)
        )
        cond_ind_test = ParCorr(significance='analytic') # 독립성 검정 방법 (선형 가정)

        # 2. PCMCIbk: NB에서 설정한 forbidden_orientation 에 대한 정보 반영
        pcmci = PCMCIbk(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=1
        )

        # 3. forbidden_orientation 적용 + run_pcmciplus
        results = pcmci.run_pcmciplus(
            tau_min=0, # 동시시점을 고려하는 경우 -> 실제 인과가 아니더라도 반영될 우려가 있음 -> tau_min을 1로 조절하는 것도 방법
            tau_max=self.tau_max,
            pc_alpha=self.sig_level,
            forbidden_orientation=self.forbidden_orientation
        )
        # results: {'graph', 'val_matrix', 'p_matrix', 'conf_matrix'...}

        # 4. graph parsing
        #    graph[i,j,t] in {"-->", "<--", "o-o", "x-x", ""}
        graph_3d = results["graph"]
        var_names = list(self.data.columns)
        n_vars = len(var_names)

        #
        for i in range(n_vars):       # 결과 변수 index
            for j in range(n_vars):   # 원인 변수 index
                if i == j:   # 본인에 대한 인과는 넘기기
                    continue

                for tau in range(self.tau_max + 1):
                    symbol = graph_3d[i, j, tau]

                    # 인과가 있는가를 확인
                    if symbol == '-->':
                        cause_name  = var_names[i]
                        effect_name = var_names[j]
                        lag_val = -tau
                        # window_causal_graph_dict[effect_name]에 추가
                        self.window_causal_graph_dict[effect_name].append(
                            (cause_name, lag_val)
                        )

        print("=== [CB] final window_causal_graph_dict ===")
        for var in var_names:
            print(f"{var}: {self.window_causal_graph_dict[var]}")

    def run(self):
        self._noise_based()
        self._constraint_based()

        target_var = "Com_Gold"
        if target_var in self.window_causal_graph_dict:
            com_gold_causes = [cause for (cause, lag) in self.window_causal_graph_dict[target_var]]
        else:
            com_gold_causes = []

        return self.window_causal_graph_dict, com_gold_causes
