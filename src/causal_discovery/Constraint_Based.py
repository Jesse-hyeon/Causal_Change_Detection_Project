import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.jpcmciplus import JPCMCIplus

class pcmciplus_model:
    def __init__(self, data: pd.DataFrame, tau_max=3, alpha=0.05, target_var="Com_Gold"):
        self.data = data
        self.tau_max = tau_max
        self.alpha = alpha
        self.target_var = target_var

    def select_features_pcmci_plus(self):
        # 1. 데이터 변환
        dataframe = pp.DataFrame(self.data.values, var_names=list(self.data.columns))

        # 2. 독립성 검정 함수
        ind_test = ParCorr(significance='analytic')

        # 3. 변수 역할 지정
        node_classification = {i: "system" for i in range(self.data.shape[1])}

        # 4. PCMCI+ 구성
        pcmci_plus = JPCMCIplus(
            dataframe=dataframe,
            cond_ind_test=ind_test,
            node_classification=node_classification # 이거 없으면 에러떠서 추가함
        )

        # 5. run_pcmciplus 실행
        results_plus = pcmci_plus.run_pcmciplus(
            tau_min=0,
            tau_max=self.tau_max,
            pc_alpha=self.alpha
        )

        # 6. 유의미한 인과간선 출력
        pcmci_plus.print_significant_links(
            p_matrix=results_plus['p_matrix'],
            val_matrix=results_plus['val_matrix'],
            alpha_level=self.alpha
        )

        # 7. target 변수에 대한 feature selection
        var_names = list(self.data.columns)
        target_idx = var_names.index(self.target_var)

        p_matrix = results_plus['p_matrix']
        causal_features = []

        for col_idx, col_name in enumerate(var_names):
            if col_name == self.target_var:
                continue
            for tau in range(self.tau_max + 1):
                pval = p_matrix[target_idx, col_idx, tau]
                if pval < self.alpha:
                    causal_features.append((col_name, -tau))

        return causal_features
