import sys
import os
sys.path.append(os.path.abspath("src"))

import copy
# from types import SimpleNamespace

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit

from sklearn.model_selection import BaseCrossValidator
from train.exp_former import Exp_Main_former
from train.exp_rnn import Exp_Main_rnn

### sliding window split
class FixedPredLenSplit(BaseCrossValidator):
    def __init__(self, n_splits, pred_len):
        self.n_splits = n_splits  # 교차 검증 Fold 개수
        self.pred_len = pred_len  # Validation 크기 고정 (pred_len)

    def split(self, X, y=None, groups=None):
        """
        데이터를 (train, validation)으로 나누는 함수
        """
        n_samples = len(X)
        indices = np.arange(n_samples)  # 전체 데이터 인덱스

        for split in range(self.n_splits):
            train_end = n_samples - self.pred_len * (self.n_splits - split)  # Train 데이터 끝 인덱스
            val_start = train_end  # Validation 시작 인덱스
            val_end = val_start + self.pred_len  # Validation 끝 인덱스 (고정 크기)

            yield indices[:train_end], indices[val_start:val_end]

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits

### optuna class
class HyperParameterTuner:
    def __init__(self, base_config, param_ranges, n_splits=3, n_trials=50):
        """
        Parameters:
            base_config (dict): 고정 설정 값 (데이터, 모델 기본 설정 등)
            param_ranges (dict): 튜닝할 하이퍼파라미터의 범위 및 타입 정보
            n_splits (int): TS‑CV에서 사용할 fold 수
            n_trials (int): Optuna에서 시도할 trial 수
        """
        self.base_config = base_config
        self.param_ranges = param_ranges
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.best_params = None

    def _train_and_evaluate(self, args, setting="trial_experiment"):
        """
        ExpMain 객체를 생성하여 학습을 진행한 후, validation 데이터에 대한 loss(MSE)를 반환합니다.
        """
        if args["model"] in args['former_model']:
            trainer = Exp_Main_former(args)
        else:
            trainer = Exp_Main_rnn(args)
        trainer.train(setting=setting)
        vali_data, vali_loader = trainer._get_data(flag='val')
        criterion = trainer._select_criterion()
        val_loss = trainer.vali(vali_data, vali_loader, criterion)
        return val_loss

    def _run_trial(self, args):
        """
        TimeSeriesSplit을 이용하여, 각 fold마다 모델을 학습/평가하고,
        fold별 validation loss의 평균을 반환합니다.

        주의: data_provider 함수 또는 ExpMain 클래스가 args 내에
              train_indices, val_indices를 활용하여 해당 인덱스의 데이터만 반환하도록 구현되어 있어야 합니다.
        """
        # 전체 데이터셋 불러오기 (train flag)
        train_len = args["num_train_original"]
        total_indices = np.arange(train_len)

        # TimeSeriesSplit을 사용하여 train/val 분할
        tscv = FixedPredLenSplit(n_splits=self.n_splits, pred_len=args["pred_len"])
        fold_losses = []
        fold_id = 0

        for train_idx, val_idx in tscv.split(total_indices):
            fold_id += 1
            print(f"--- TS-CV Fold {fold_id}/{self.n_splits} ---")
            # args를 복제하고, fold별 train/val 인덱스 정보를 추가
            fold_args = copy.deepcopy(args)
            fold_args["num_train"] = len(train_idx)
            fold_args["num_valid"] = len(val_idx)

            fold_loss = self._train_and_evaluate(fold_args, setting=f"trial_fold_{fold_id}")
            print(f"Fold {fold_id} validation loss: {fold_loss:.4f}")
            fold_losses.append(fold_loss)

        avg_loss = np.mean(fold_losses)
        print(f"Average validation loss over {self.n_splits} folds: {avg_loss:.4f}")
        return avg_loss


    def objective(self, trial):
        """
        Optuna Objective 함수.
        샘플링된 하이퍼파라미터를 base_config와 병합하여 TS‑CV 평가를 진행하고,
        평균 validation loss를 반환합니다.
        """
        sampled_params = {}
        for param, config in self.param_ranges.items():
            if config.get("type") == "int":
                step = config.get("step", 1)
                sampled_params[param] = trial.suggest_int(param, config["low"], config["high"], step=step)
            elif config.get("type") == "float":
                if config.get("log", False):
                    sampled_params[param] = trial.suggest_float(param, config["low"], config["high"], log=True)
                else:
                    sampled_params[param] = trial.suggest_float(param, config["low"], config["high"])
            elif config.get("type") == "categorical":
                sampled_params[param] = trial.suggest_categorical(param, config["choices"])

        # base_config와 sampled_params 병합
        config = self.base_config.copy()
        config.update(sampled_params)
        args = config

        # TS‑CV 기반 평가 실행
        val_loss = self._run_trial(args)
        return val_loss

    def run_study(self):
        """
        Optuna Study를 실행하여 최적의 하이퍼파라미터를 탐색합니다.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        print("Best hyperparameters:", study.best_params)
        self.best_params = study.best_params
        return study

    def train_final_model(self):
        """
        최적의 하이퍼파라미터(best_params)를 바탕으로 전체 training set으로 모델을 재학습하고,
        test set에 대해 최종 평가를 진행합니다.
        """
        print("Final training model")
        final_config = self.base_config.copy()
        final_config.update(self.best_params)
        final_config["num_train"] = final_config["num_train_original"]
        final_config["num_valid"] = final_config["num_valid_original"]

        final_args = final_config

        if final_args["model"] in final_args['former_model']:
            trainer = Exp_Main_former(final_args)
        else:
            trainer = Exp_Main_rnn(final_args)
        trainer.train(setting="final_experiment")
        trainer.test(setting="final_experiment", test=0)
        print("Final model training and testing complete.")