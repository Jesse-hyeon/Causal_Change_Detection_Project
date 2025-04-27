import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

class lasso_model():
    def __init__(self, config=None, alpha=0.1, max_iter=10000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = None
        self.fitted = False
        self.selected_features_ = None
        self.coefficients_ = None
        self.best_alpha_ = None
        self.config = config

    def preprocess(self, X):
        X = X.copy()
        if 'date' in X.columns:
            X['date'] = pd.to_datetime(X['date'], errors='coerce')
            X['year'] = X['date'].dt.year
            X['month'] = X['date'].dt.month
            X['dayofweek'] = X['date'].dt.dayofweek
            X = X.drop(columns='date')
        return X

    def fit(self, X_train, y_train):
        X_train = self.preprocess(X_train)

        if self.alpha == 'auto':
            self.model = LassoCV(cv=5, max_iter=self.max_iter)
            self.model.fit(X_train, y_train)
            self.alpha = self.model.alpha_
            self.best_alpha_ = self.model.alpha_
        else:
            self.model = Lasso(alpha=self.alpha, max_iter=self.max_iter)
            self.model.fit(X_train, y_train)

        self.fitted = True
        self.selected_features_ = self.model.feature_names_in_[self.model.coef_ != 0]
        self.coefficients_ = pd.Series(self.model.coef_, index=self.model.feature_names_in_)

    def predict(self, X):
        if not self.fitted:
            raise Exception("Model is not fitted yet. Call 'fit' first.")
        X = self.preprocess(X)
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return rmse

    def get_selected_features(self):
        if not self.fitted:
            raise Exception("Model is not fitted yet. Call 'fit' first.")
        return self.selected_features_

    def get_coefficients(self):
        if not self.fitted:
            raise Exception("Model is not fitted yet. Call 'fit' first.")
        return self.coefficients_

    def get_best_alpha(self):
        if self.best_alpha_ is None:
            raise Exception("Alpha was not selected automatically or model not fitted yet.")
        return self.best_alpha_

    def filter_features_by_multicollinearity(self, X_train, vif_thresh=5.0, corr_thresh=0.8):
        """
        get_selected_features() 결과를 기반으로 다중공선성 제거.
        """
        selected = list(self.get_selected_features())
        coef_map = {k: abs(v) for k, v in self.get_coefficients().items() if k in selected}
        X_selected = self.preprocess(X_train)[selected]

        vif_vals = [variance_inflation_factor(X_selected.values, i) for i in range(X_selected.shape[1])]
        vif_df = pd.DataFrame({"var": X_selected.columns, "vif": vif_vals})
        high_vif = vif_df[vif_df["vif"] > vif_thresh]["var"].tolist()

        corr_matrix = X_selected[high_vif].corr().abs()
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

        selected_final = []
        selected_set = set()

        for group in groups:
            best = max(group, key=lambda var: abs(coef_map.get(var, 0)))
            if best not in selected_set:
                selected_final.append(best)
                selected_set.add(best)

        ungrouped = set(selected) - set().union(*groups)
        for var in ungrouped:
            if var not in selected_set:
                selected_final.append(var)
                selected_set.add(var)

        return sorted(selected_final)