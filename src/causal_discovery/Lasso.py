import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

class lasso_model():
    def __init__(self, alpha=0.1, max_iter=10000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = Lasso(alpha=self.alpha, max_iter=self.max_iter)
        self.fitted = False
        self.selected_features_ = None
        self.coefficients_ = None

    def preprocess(self, X):
        X = X.copy()
        if 'date' in X.columns:
            X['date'] = pd.to_datetime(X['date'], errors='coerce')  # 문자열 → datetime
            X['year'] = X['date'].dt.year
            X['month'] = X['date'].dt.month
            X['dayofweek'] = X['date'].dt.dayofweek
            X = X.drop(columns='date')  # 원래 date는 제거
        return X

    def fit(self, X_train, y_train):
        X_train = self.preprocess(X_train)
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