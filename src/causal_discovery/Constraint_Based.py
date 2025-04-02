import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

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
