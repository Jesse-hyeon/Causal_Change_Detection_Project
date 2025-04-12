import copy
import numpy as np
import optuna
from sklearn.model_selection import BaseCrossValidator
from src.train.exp_former import Exp_Main_former
from src.train.exp_rnn import Exp_Main_rnn
from optuna.samplers import GridSampler, RandomSampler

class FixedPredLenSplit(BaseCrossValidator):
    def __init__(self, n_splits, pred_len):
        self.n_splits = n_splits
        self.pred_len = pred_len

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)

        for split in range(self.n_splits):
            train_end = n_samples - self.pred_len * (self.n_splits - split)
            val_start = train_end
            val_end = val_start + self.pred_len
            yield indices[:train_end], indices[val_start:val_end]

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits

class HyperParameterTuner:
    def __init__(self, base_config, param_ranges, n_splits=3, n_trials=50):
        self.base_config = base_config
        self.param_ranges = param_ranges
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.best_params = None

    def _train_and_evaluate(self, args, setting="trial_experiment"):
        if args["model"] in args['former_model']:
            trainer_initial = Exp_Main_former(args, final_run=False)
        else:
            trainer_initial = Exp_Main_rnn(args, final_run=False)
        trainer_initial.train(setting=setting)
        vali_data, vali_loader = trainer_initial._get_data(flag='val')
        criterion = trainer_initial._select_criterion()
        val_loss = trainer_initial.vali(vali_data, vali_loader, criterion)
        return val_loss

    def _run_trial(self, args):
        train_len = args["num_train_original"]
        total_indices = np.arange(train_len)
        tscv = FixedPredLenSplit(n_splits=self.n_splits, pred_len=args["pred_len"])
        fold_losses = []
        fold_id = 0

        for train_idx, val_idx in tscv.split(total_indices):
            fold_id += 1
            print(f"--- TS-CV Fold {fold_id}/{self.n_splits} ---")
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
        if self.base_config.get("use_randomsearch", False):
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
        else:
            sampled_params = {param: trial.suggest_categorical(param, values)
                              for param, values in self.param_ranges.items()}

        print(f"[Trial {trial.number}] Sampled Hyperparameters:")
        for k, v in sampled_params.items():
            print(f"  - {k}: {v}")

        config = self.base_config.copy()
        config.update(sampled_params)
        return self._run_trial(config)

    def run_study(self):
        if self.base_config.get("use_randomsearch", False):
            sampler = RandomSampler(seed=self.base_config.get("seed", 42))
        else:
            sampler = GridSampler(self.param_ranges)

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(self.objective, n_trials=self.n_trials)
        print("Best hyperparameters:", study.best_params)
        self.best_params = study.best_params
        return study

    def train_final_model(self):
        print("Final training model")
        final_config = self.base_config.copy()
        final_config.update(self.best_params)
        final_config["num_train"] = final_config["num_train_original"]
        final_config["num_valid"] = final_config["num_valid_original"]

        final_args = final_config

        if final_args["model"] in final_args['former_model']:
            trainer_final = Exp_Main_former(final_args, final_run=True)
        else:
            trainer_final = Exp_Main_rnn(final_args, final_run=True)
        trainer_final.train(setting="final_experiment")
        trainer_final.test(setting="final_experiment", test=0)
        print("Final model training and testing complete.")
