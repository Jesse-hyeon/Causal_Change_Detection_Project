import numpy as np
import pandas as pd

# Tigramite + background knowledge 버전
# from tigramite.pcmci import PCMCI             # <- 순정 PCMCI
from src.utils.pcmci_with_bk import PCMCI as PCMCIbk # <- forbidden_orientation 지원 버전, 논문에서 사용한 것
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr

# lingam 라이브러리
from lingam import VARLiNGAM

class NBCBw:
    """
    Noise-Based (VarLiNGAM) + Constraint-Based (PCMCI+ with background knowledge) 시계열 인과 모델
    1) Noise-based에서 lag별 인과 계수행렬로 forbidden_orientation 설정
    2) Constraint-based(PCMCI+)에서 forbidden_orientation을 적용해 뒤집지 않도록 제약
    3) 최종 window_causal_graph_dict에는 원인에 대한 (변수명, -lag) 정보를 담는다
    """

    def __init__(self,
                 data: pd.DataFrame,
                 tau_max: int = 3,
                 sig_level: float = 0.05,
                 threshold: float = 0.01,
                 linear: bool = True):
        """
        :param data:      시계열 DataFrame (행=시간, 열=변수)
        :param tau_max:   최대 시차(lag) 설정 (Noise-based와 PCMCI 모두 사용)
        :param sig_level: PCMCI+에서 사용될 유의수준(pc_alpha)
        :param threshold: VarLiNGAM 계수 필터링 임계값
        :param linear:    True면 VarLiNGAM(선형), False면 다른 Noise-based 가능(RESIT 등)
        """
        self.data = data
        self.tau_max = tau_max
        self.sig_level = sig_level
        self.threshold = threshold
        self.linear = linear

        # Noise-Based 결과
        self.forbidden_orientation = []         # (effect_idx, cause_idx) 튜플
        self.window_causal_graph_dict = {
            col: [] for col in self.data.columns
        }

    def _noise_based(self):
        """
        Noise-Based: VarLiNGAM(시차=lags) → 인과 계수행렬(adjacency_matrices_) 활용
        - 절댓값이 threshold 초과인 계수만 의미있는 인과로 판단
        - (effect_idx, cause_idx)를 forbidden_orientation에 기록(반대방향 금지)
        - window_causal_graph_dict에 (원인변수명, -lag) 저장
        """
        print("=== [NBCBw] Noise-Based Step: VarLiNGAM ===")
        var_names = list(self.data.columns)
        model = VARLiNGAM(lags=self.tau_max, criterion='bic', prune=False)
        model.fit(self.data.values)

        adjacency_mats = model.adjacency_matrices_  # [array(lag1), array(lag2), ...]
        n_vars = len(var_names)

        # lag=1 => j(t-1)->i(t), lag=2 => j(t-2)->i(t), ...
        for lag_idx, mat in enumerate(adjacency_mats, start=1):
            for i in range(n_vars):       # effect
                for j in range(n_vars):   # cause
                    coef = mat[i, j]
                    if abs(coef) > self.threshold:
                        # j(t-lag_idx) --> i(t)
                        effect_name = var_names[i]
                        cause_name  = var_names[j]

                        # forbidden_orientation: i->j는 뒤집으면 안 됨 => (i, j)
                        self.forbidden_orientation.append((i, j))

                        # window_causal_graph_dict: (cause, -lag_idx)
                        self.window_causal_graph_dict[effect_name].append(
                            (cause_name, -lag_idx)
                        )

        print("### [NB] forbidden_orientation:", self.forbidden_orientation)
        for var in self.window_causal_graph_dict:
            print(f"{var}: {self.window_causal_graph_dict[var]}")

    def _constraint_based(self):
        """
        Constraint-Based(PCMCI+ with BK):
          - forbidden_orientation로 NB 단계에서 확정된 방향을 뒤집지 않도록
            links_for_pc 후보에서 제거 → run_pcmciplus 실행
          - PCMCI+ 결과에서 graph 파싱해서, window_causal_graph_dict를 업데이트
        """
        print("=== [NBCBw] Constraint-Based Step: PCMCI+ ===")

        # 1. Tigramite용 data & ParCorr
        dataframe = pp.DataFrame(
            data=self.data.values,
            var_names=list(self.data.columns)
        )
        cond_ind_test = ParCorr(significance='analytic')

        # 2. PCMCIbk: 금지방향 인자를 받을 수 있는 버전
        pcmci = PCMCIbk(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=1
        )

        # 3. forbidden_orientation 적용 + run_pcmciplus
        #    forbidden_orientation = [(effect_idx, cause_idx), ...]
        results = pcmci.run_pcmciplus(
            tau_min=0,
            tau_max=self.tau_max,
            pc_alpha=self.sig_level,
            forbidden_orientation=self.forbidden_orientation
        )
        # results: {'graph', 'val_matrix', 'p_matrix', 'conf_matrix'...}

        # 4. graph parsing (string array)
        #    graph[i,j,t] in {"-->", "<--", "o-o", "x-x", ""}
        graph_3d = results["graph"]
        var_names = list(self.data.columns)
        n_vars = len(var_names)

        # 주의: PCMCI+가 lag=0 ~ tau_max까지 모두 처리
        # j(t)는 i(t - tau) link
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                for tau in range(self.tau_max + 1):
                    symbol = graph_3d[i, j, tau]
                    if symbol == '-->':
                        # i(t - tau) => j(t)
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
        """
        최종 실행:
        1) noise-based (VarLiNGAM)
        2) constraint-based (PCMCI+ with forbidden_orientation)
        3) Com_Gold에 영향을 미치는 변수만 별도 리스트로 반환 (원하는 변수로 교체 가능)
        """
        self._noise_based()
        self._constraint_based()

        target_var = "Com_Gold"
        if target_var in self.window_causal_graph_dict:
            com_gold_causes = [cause for (cause, lag) in self.window_causal_graph_dict[target_var]]
        else:
            com_gold_causes = []

        return self.window_causal_graph_dict, com_gold_causes
