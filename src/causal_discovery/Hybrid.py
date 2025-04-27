import networkx as nx
import pandas as pd
import numpy as np
import tigramite.data_processing as pp

from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from lingam.var_lingam import VARLiNGAM
from lingam.resit import RESIT
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_varlingam(data, tau_max):
    model = VARLiNGAM(lags=tau_max, criterion='bic', prune=True)
    model.fit(data.values)
    order = model.causal_order_
    col_names = list(data.columns)
    order = [col_names[i] for i in order]
    order.reverse()

    order_matrix = pd.DataFrame(np.zeros((data.shape[1], data.shape[1])),
                                columns=col_names, index=col_names, dtype=int)
    for col_i in order_matrix.index:
        for col_j in order_matrix.columns:
            if col_i != col_j:
                idx_i = order.index(col_i)
                idx_j = order.index(col_j)
                if idx_i > idx_j:
                    order_matrix.loc[col_j, col_i] = 2  # 원래 코드에서는 order_matrix[col_j].loc[col_i] = 2
                    order_matrix.loc[col_i, col_j] = 1  # 원래 코드에서는 order_matrix[col_i].loc[col_j] = 1
    return order_matrix

class CBNBe:
    def __init__(self, data, tau_max, sig_level=0.05, linear=True,
                 model="linear", indtest="linear", cond_indtest="linear"):
        self.data = data
        self.tau_max = tau_max
        self.sig_level = sig_level
        self.linear = linear
        self.model = model
        self.indtest = indtest
        self.cond_indtest = cond_indtest

        self.col_names = list(data.columns)
        d = len(self.col_names)
        self.window_causal_graph = np.full((d, d, tau_max + 1), "---", dtype=object)
        self.window_causal_graph_dict = {col: [] for col in self.col_names}
        self.causal_graph = GeneralGraph([GraphNode(col) for col in self.col_names])

    def constraint_based(self):
        print("######## Running Constraint-based (PCMCI) ########")
        df = pp.DataFrame(data=self.data.values, var_names=self.col_names)
        cond_ind_test = ParCorr(significance='analytic') if self.linear else NotImplementedError(
            "Non-linear not implemented.")
        pcmci = PCMCI(dataframe=df, cond_ind_test=cond_ind_test)
        results = pcmci.run_pcmci(tau_min=0, tau_max=self.tau_max, pc_alpha=self.sig_level)
        skeleton = results["graph"]
        d = len(self.col_names)
        self.window_causal_graph = skeleton.copy()

    def find_cycle_groups(self):
        print("######## find_cycle_groups for 0-lag edges ########")
        d = len(self.col_names)
        G0 = nx.Graph()
        for i in range(d):
            for j in range(i + 1, d):
                if self.window_causal_graph[i, j, 0] in ["o-o", "x-x", "-->", "<--"]:
                    G0.add_edge(self.col_names[i], self.col_names[j])
        list_cycles = nx.cycle_basis(G0)
        cycle_groups = {}
        idx = 0
        for cyc in list_cycles:
            cyc = sorted(list(cyc))
            merged = False
            for k in cycle_groups:
                if len(set(cycle_groups[k]).intersection(cyc)) >= 2:
                    cycle_groups[k] = list(set(cycle_groups[k]).union(cyc))
                    merged = True
                    break
            if not merged:
                cycle_groups[idx] = cyc
                idx += 1
        instantaneous_nodes = list(G0.nodes())
        print("[DEBUG] found cycles:", list_cycles)
        return cycle_groups, list_cycles, instantaneous_nodes

    def noise_based(self):
        print("######## Running Noise-based ########")
        cycle_groups, list_cycles, instantaneous_nodes = self.find_cycle_groups()
        print("[DEBUG] cycle_groups:", cycle_groups)
        print("[DEBUG] instantaneous_nodes:", instantaneous_nodes)
        if len(instantaneous_nodes) < 2:
            return
        for group in cycle_groups.values():
            parents = list(set(self.col_names) - set(group))
            sub_data = self.data[group + parents]
            try:
                model = VARLiNGAM(lags=self.tau_max, criterion='bic', prune=True)
                model.fit(sub_data.values)
            except Exception as e:
                print(f"[ERROR] VARLiNGAM failed: {e}")
                continue
            order_idx = model.causal_order_
            sub_cols = sub_data.columns.tolist()
            ordered_vars = [sub_cols[i] for i in order_idx]
            for xi in group:
                for xj in group:
                    if xi != xj:
                        if ordered_vars.index(xi) > ordered_vars.index(xj):
                            i = self.col_names.index(xi)
                            j = self.col_names.index(xj)
                            if self.window_causal_graph[i, j, 0] in ["o-o", "x-x", "-->", "<--"]:
                                self.window_causal_graph[i, j, 0] = "<--"
                                self.window_causal_graph[j, i, 0] = "-->"
                                print(f"[noise_based orientation] {xj} -> {xi} (lag=0)")

    def construct_summary_causal_graph(self):
        print("######## Construct summary causal graph ########")
        d = len(self.col_names)
        summary_matrix = pd.DataFrame(np.zeros((d, d)), columns=self.col_names, index=self.col_names)
        for i in range(d):
            for j in range(d):
                for t in range(self.tau_max + 1):
                    if self.window_causal_graph[i, j, t] == "-->":
                        src, dst = self.col_names[i], self.col_names[j]
                        if (src, -t) not in self.window_causal_graph_dict[dst]:
                            self.window_causal_graph_dict[dst].append((src, -t))
                            summary_matrix.loc[src, dst] = 1
                    elif self.window_causal_graph[i, j, t] == "<--":
                        src, dst = self.col_names[i], self.col_names[j]
                        if (dst, -t) not in self.window_causal_graph_dict[src]:
                            self.window_causal_graph_dict[src].append((dst, -t))
                            summary_matrix.loc[dst, src] = 1
        for i in range(d):
            for j in range(d):
                if i != j:
                    src, dst = self.col_names[i], self.col_names[j]
                    if summary_matrix.loc[src, dst] == 1 and summary_matrix.loc[dst, src] == 1:
                        self.causal_graph.add_edge(Edge(GraphNode(src), GraphNode(dst), Endpoint.ARROW, Endpoint.ARROW))
                    elif summary_matrix.loc[src, dst] == 1:
                        self.causal_graph.add_edge(Edge(GraphNode(src), GraphNode(dst), Endpoint.TAIL, Endpoint.ARROW))
                    elif summary_matrix.loc[dst, src] == 1:
                        self.causal_graph.add_edge(Edge(GraphNode(dst), GraphNode(src), Endpoint.TAIL, Endpoint.ARROW))

    def filter_cbnb_by_multicollinearity(data: pd.DataFrame, causal_list: list, vif_thresh=5.0, corr_thresh=0.8):
        """
        CBNB의 causal_list [(var, lag), ...] 를 입력받아
        VIF + 상관관계 기반으로 다중공선성 제거하고
        중요 변수만 반환
        """
        # Step 1: 변수별 인과 횟수(가중치) 계산
        from collections import Counter
        var_counter = Counter([var for var, lag in causal_list])

        # Step 2: VIF 계산
        vars_unique = sorted(set(var_counter.keys()))
        X = data[vars_unique].copy()

        vif_vals = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_df = pd.DataFrame({"var": X.columns, "vif": vif_vals})
        high_vif = vif_df[vif_df["vif"] > vif_thresh]["var"].tolist()

        # Step 3: 상관관계 그룹핑
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

        # Step 4: 각 그룹에서 등장횟수(인과점수) 가장 높은 변수 선택
        selected = []
        for group in groups:
            best = max(group, key=lambda var: var_counter.get(var, 0))
            selected.append(best)

        # Step 5: 그룹 밖 변수는 그대로 추가
        ungrouped = set(vars_unique) - set().union(*groups)
        selected += list(ungrouped)

        return sorted(selected)

    def run(self):
        self.constraint_based()
        print(self.window_causal_graph, "(after constraint_based)")
        self.noise_based()
        print(self.window_causal_graph, "(after noise_based)")
        self.construct_summary_causal_graph()
        print("[Final] window_causal_graph_dict:", self.window_causal_graph_dict)

class NBCBe:
    def __init__(self, data, tau_max, sig_level=0.05, linear=True,
                 model="linear", indtest="linear", cond_indtest="linear"):
        self.data = data
        self.tau_max = tau_max
        self.sig_level = sig_level
        self.linear = linear
        self.model = model
        self.indtest = indtest
        self.cond_indtest = cond_indtest

        self.col_names = list(data.columns)
        self.order_matrix = None # 깃허브 코드에서 causal_order과 동일한 부분
        self.forbidden_orientation = []
        self.window_causal_graph_dict = {col: [] for col in self.col_names}
        self.causal_graph = GeneralGraph([GraphNode(col) for col in self.col_names])

    def noise_based(self):
        if self.linear:
            self.order_matrix = run_varlingam(self.data, self.tau_max)
        else: #여기 비선형 안쓰면 그냥 바꾸기
            self.order_matrix = run_resit(self.data)

        print("Order Matrix from VarLiNGAM:")
        print(self.order_matrix)

        #  self.order_matrix가 causal_order과 동일한 역할을 함
        for i in self.col_names:
            for j in self.col_names:
                if i != j:
                    if self.order_matrix.loc[i, j] == 2 and self.order_matrix.loc[j, i] == 1:
                        src_index = self.col_names.index(j)
                        dst_index = self.col_names.index(i)
                        if (src_index, dst_index) not in self.forbidden_orientation:
                            self.forbidden_orientation.append((src_index, dst_index))
        print("Forbidden Orientations (from VarLiNGAM):", self.forbidden_orientation)

    def constraint_based(self):
        print("######## Running Constraint-based (PCMCI) ########")
        df = pp.DataFrame(data=self.data.values, var_names=self.col_names)
        cond_ind_test = ParCorr(significance='analytic')
        pcmci = PCMCI(dataframe=df, cond_ind_test=cond_ind_test)
        results = pcmci.run_pcmci(tau_min=0, tau_max=self.tau_max, pc_alpha=self.sig_level)
        skeleton = results["graph"]

        summary_matrix = pd.DataFrame(np.zeros((self.data.shape[1], self.data.shape[1])),
                                      index=self.data.columns, columns=self.data.columns, dtype=int)
        for i in range(len(self.col_names)):
            for j in range(len(self.col_names)):
                if skeleton[i, j, 0] in ["o-o", "x-x", "-->", "<--"]:
                    summary_matrix.iloc[i, j] = 1

        for col_i in self.data.columns:
            for col_j in self.data.columns:
                if col_i != col_j and summary_matrix.loc[col_i, col_j] == 1 and summary_matrix.loc[col_j, col_i] == 1:
                    if self.order_matrix.loc[col_i, col_j] == 1:
                        summary_matrix.loc[col_j, col_i] = 0
                    elif self.order_matrix.loc[col_i, col_j] == 2:
                        summary_matrix.loc[col_i, col_j] = 0

        print("Summary Matrix after background knowledge adjustment:")
        print(summary_matrix)

        # summary_matrix를 바탕으로 causallearn 그래프 구성
        for col_i in self.data.columns:
            for col_j in self.data.columns:
                if col_i != col_j:
                    if summary_matrix.loc[col_i, col_j] == 1 and summary_matrix.loc[col_j, col_i] == 1:
                        self.causal_graph.add_edge(
                            Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.ARROW, Endpoint.ARROW)
                        )
                    elif summary_matrix.loc[col_i, col_j] == 1:
                        self.causal_graph.add_edge(
                            Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.TAIL, Endpoint.ARROW)
                        )
                    elif summary_matrix.loc[col_j, col_i] == 1:
                        self.causal_graph.add_edge(
                            Edge(GraphNode(col_j), GraphNode(col_i), Endpoint.TAIL, Endpoint.ARROW)
                        )

        for src in self.data.columns:
            for dst in self.data.columns:
                if src != dst and summary_matrix.loc[src, dst] == 1:
                    self.window_causal_graph_dict[dst].append((src, 0))
        print("Window Causal Graph Dict:")
        print(self.window_causal_graph_dict)

    def filter_nbcb_by_multicollinearity(data: pd.DataFrame, causal_list: list, vif_thresh=5.0, corr_thresh=0.8):
        """
        CBNB의 causal_list [(var, lag), ...] 를 입력받아
        VIF + 상관관계 기반으로 다중공선성 제거하고
        중요 변수만 반환
        """
        # Step 1: 변수별 인과 횟수(가중치) 계산
        from collections import Counter
        var_counter = Counter([var for var, lag in causal_list])

        # Step 2: VIF 계산
        vars_unique = sorted(set(var_counter.keys()))
        X = data[vars_unique].copy()

        vif_vals = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_df = pd.DataFrame({"var": X.columns, "vif": vif_vals})
        high_vif = vif_df[vif_df["vif"] > vif_thresh]["var"].tolist()

        # Step 3: 상관관계 그룹핑
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

        # Step 4: 각 그룹에서 등장횟수(인과점수) 가장 높은 변수 선택
        selected = []
        for group in groups:
            best = max(group, key=lambda var: var_counter.get(var, 0))
            selected.append(best)

        # Step 5: 그룹 밖 변수는 그대로 추가
        ungrouped = set(vars_unique) - set().union(*groups)
        selected += list(ungrouped)

        return sorted(selected)

    def run(self):
        self.noise_based()
        print(self.window_causal_graph_dict, "(after noise_based)")
        self.constraint_based()
        print(self.window_causal_graph_dict, "(after constraint_based)")
        print("Final Causal Graph:", self.causal_graph)