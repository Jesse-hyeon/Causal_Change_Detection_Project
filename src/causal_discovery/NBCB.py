import numpy as np
import pandas as pd

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from lingam import VARLiNGAM

class NBCBw:
    """
    Noise-Based (VarLiNGAM) + Constraint-Based (PCMCI+) 시계열 인과추론 모델
    - Window Causal Graph 형태 (lag>0은 시간방향, lag=0은 noise-based 순서)
    - forbidden_orientation: NB 단계에서 결정한 인과방향을
      CB 단계에서 뒤집지 못하도록 하는 제약
    """

    def __init__(self,
                 data: pd.DataFrame,
                 tau_max: int = 3,
                 sig_level: float = 0.05,
                 linear: bool = True):
        """
        :param data: 시계열 DataFrame (행=시간, 열=변수)
        :param tau_max: 최대 시차(lag) 설정
        :param sig_level: 유의수준(alpha) 예: 0.05
        :param linear: True면 VarLiNGAM 사용 (비선형 X),
                       False면 여유롭게 RESIT 등 대안 사용 가능
        """
        self.data = data
        self.tau_max = tau_max
        self.sig_level = sig_level
        self.linear = linear

        # Noise-Based 결과
        self.causal_order_matrix = None       # VarLiNGAM에서 추출된 인과순서 매트릭스
        self.forbidden_orientation = []       # NB 단계에서 금지된 방향 (i->j 고정)
        self.window_causal_graph_dict = {col: [] for col in data.columns}

    def _noise_based(self):
        """
        VarLiNGAM 등을 이용해 동시시점 인과순서(순위) 먼저 추정
        - causal_order_matrix (상삼각=2, 하삼각=1)
        - forbidden_orientation (뒤집지 못할 방향 목록)
        """
        print("=== [NBCBw] Noise-Based Step: VarLiNGAM ===")

        # 1. VarLiNGAM 모델 적합
        model = VARLiNGAM(lags=3, criterion='bic', prune=False) if self.linear else None
        model.fit(self.data.values)

        order_list = model.causal_order_  # 예: [0, 2, 1, ...] (변수 인덱스 순서)
        var_names = list(self.data.columns)
        n = len(var_names)

        # 2. 인과순서 행렬
        order_matrix = pd.DataFrame(
            np.zeros((n, n)),
            columns=var_names,
            index=var_names,
            dtype=int
        )

        # order_list에서 앞쪽일수록 “원인”에 가깝다
        for i in range(n):
            for j in range(i + 1, n):
                idx_i = order_list.index(i)
                idx_j = order_list.index(j)
                if idx_i < idx_j:
                    # i -> j
                    order_matrix.iloc[i, j] = 1
                    order_matrix.iloc[j, i] = 2
                else:
                    # j -> i
                    order_matrix.iloc[j, i] = 1
                    order_matrix.iloc[i, j] = 2

        self.causal_order_matrix = order_matrix

        # 3. forbidden_orientation 설정
        for i in range(n):
            for j in range(n):
                if i != j and order_matrix.iloc[i, j] == 1 and order_matrix.iloc[j, i] == 2:
                    # i->j가 결정됐으니 j->i는 금지
                    self.forbidden_orientation.append((j, i))

        print("Causal Order Matrix:\n", self.causal_order_matrix)

    def _constraint_based(self):
        """
        PCMCI+ (Constraint-Based) 단계
        - forbidden_orientation을 이용해 NBCB에서 뒤집으면 안되는 방향을 보존
        """
        print("=== [NBCBw] Constraint-Based Step: PCMCI+ ===")

        # 1. Tigramite용 DataFrame 생성
        dataframe = pp.DataFrame(
            data=self.data.values,
            var_names=list(self.data.columns)
        )

        # 2. 독립성 검정(ParCorr) + PCMCI+
        cond_ind_test = ParCorr(significance='analytic')

        # 3. PCMCI + 실행
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
        results = pcmci.run_pcmciplus(tau_min=0, tau_max=self.tau_max, pc_alpha=self.sig_level)

        if isinstance(results, tuple):
            results = results[0]
        graph_3d = np.squeeze(results["graph"], axis=-1) if results["graph"].ndim == 4 else results["graph"]

        # 4. 결과 배열 확인
        print("PCMCI+ graph shape:", results["graph"].shape)

        # 5. 결과 파싱
        n = self.data.shape[1]
        var_names = list(self.data.columns)
        for i in range(n):
            for j in range(n):
                for tau in range(self.tau_max + 1):
                    if i == j:
                        continue
                    symbol = graph_3d[i, j, tau]

                    # 6. 예상 외의 다중 값 예외 처리
                    if isinstance(symbol, (list, tuple, np.ndarray)):
                        if len(symbol) == 1:
                            symbol = symbol[0]  # 단일 값이면 첫 번째 요소 사용
                        else:
                            print(f"⚠ Unexpected format at ({i}, {j}, {tau}):", symbol)
                            continue  # 다중 값이면 무시하고 다음으로 진행

                    # 7. 올바른 방향성인 경우 추가
                    if symbol == '-->':
                        lag_val = 0 if tau == 0 else -tau
                        cause_name = var_names[i]
                        effect_name = var_names[j]
                        self.window_causal_graph_dict[effect_name].append((cause_name, lag_val))

        print("=== [Constraint-Based] final window_causal_graph_dict ===")
        for var in var_names:
            print(var, ":", self.window_causal_graph_dict[var])

    def run(self):
        self._noise_based()
        self._constraint_based()

        # Com_Gold에 영향을 미치는 변수 리스트 저장
        com_gold_causes = [cause for cause, lag in self.window_causal_graph_dict.get("Com_Gold", [])]

        return self.window_causal_graph_dict, com_gold_causes