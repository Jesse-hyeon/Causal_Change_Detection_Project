import sys
import os

import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from src.utils.etc import EarlyStopping, adjust_learning_rate, metric
import wandb

### data
from src.train.data_custom_rnn import data_provider

### Model
from src.model.LSTM import lstm_model
from src.model.MLP import mlp_model
from src.model.RNN import rnn_model

class Exp_Basic_rnn(object):
    def __init__(self, config):
        self.config = config
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.config["use_gpu"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.config["gpu"]) if not self.config["use_multi_gpu"] else self.config["devices"]
            device = torch.device('cuda:{}'.format(self.config["gpu"]))
            print('Use GPU: cuda:{}'.format(self.config["gpu"]))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

warnings.filterwarnings('ignore')

class Exp_Main_rnn(Exp_Basic_rnn):
    def __init__(self, config, final_run):
        super(Exp_Main_rnn, self).__init__(config)
        self.final_run = final_run

    def _build_model(self):
        model_dict = {
            'lstm': lstm_model,
            'mlp': mlp_model,
            'rnn': rnn_model
        }
        model = model_dict[self.config["model"]](self.config).float()

        if self.config["use_multi_gpu"] and self.config["use_gpu"]:
            model = nn.DataParallel(model, device_ids=self.config["device_ids"])
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.config, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        all_preds = []
        all_trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # ✅ LSTM은 인코더-디코더 구조 없음
                if self.config["use_amp"]:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.config["features"] == 'MS' else 0
                outputs = outputs[:, -self.config["pred_len"]:, f_dim:]
                batch_y = batch_y[:, -self.config["pred_len"]:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
                all_preds.append(pred)
                all_trues.append(true)

        avg_loss = np.mean(total_loss)
        all_preds = torch.cat(all_preds, dim=0).numpy()  # (batch, seq, 1)
        all_trues = torch.cat(all_trues, dim=0).numpy()

        self.model.train()
        return avg_loss, all_preds, all_trues

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')

        # ✅ final_run이 아닌 경우에만 validation 데이터 로드 및 early stopping 준비
        if not self.final_run:
            vali_data, vali_loader = self._get_data(flag='val')
            path = os.path.join(self.config["checkpoints"], setting)
            if not os.path.exists(path):
                os.makedirs(path)
            early_stopping = EarlyStopping(patience=self.config["patience"], verbose=True)

        time_now = time.time()
        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.config["use_amp"]:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.config["train_epochs"]):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.config["use_amp"]:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.config["features"] == 'MS' else 0
                outputs = outputs[:, -self.config["pred_len"]:, f_dim:]
                batch_y = batch_y[:, -self.config["pred_len"]:, f_dim:]

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.config["train_epochs"] - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.config["use_amp"]:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            if not self.final_run:
                vali_loss, _, _ = self.vali(vali_data, vali_loader, criterion)
                test_loss, _, _ = self.vali(test_data, test_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))

                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            adjust_learning_rate(model_optim, epoch + 1, self.config)

        # ✅ 최종 저장 로직: final_run일 경우 저장/로드 생략
        if not self.final_run:
            best_model_path = os.path.join(self.config["checkpoints"], setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            print("✅ final_run: checkpoint 저장 및 로드 생략")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # ✅ 모델 예측
                if self.config["use_amp"]:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)  # (batch_size, pred_len, num_classes)
                else:
                    outputs = self.model(batch_x)

                # ✅ feature dimension 결정 (원본 코드 유지)
                f_dim = -1 if self.config["features"] == 'MS' else 0
                outputs = outputs[:, -self.config["pred_len"]:, f_dim:]
                batch_y = batch_y[:, -self.config["pred_len"]:, f_dim:].to(self.device)

                # ✅ CPU 변환 후 numpy 형태로 변환
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # ✅ 차원 변환 (3D -> 2D) 후 역변환
                batch_size, seq_len, feature_dim = outputs.shape  # (batch, pred_len, num_classes)
                pred_reshaped = outputs.reshape(seq_len, feature_dim)
                true_reshaped = batch_y.reshape(seq_len, feature_dim)

                pred_original = test_data.inverse_transform(pred_reshaped)
                true_original = test_data.inverse_transform(true_reshaped)

                # ✅ 다시 3D로 변환하여 리스트에 저장
                pred_original = pred_original.reshape(batch_size, seq_len, feature_dim)
                true_original = true_original.reshape(batch_size, seq_len, feature_dim)

                preds.append(pred_original)
                trues.append(true_original)

                # ✅ 이미지 시각화 및 로그
                if self.final_run and self.config.get("use_wandb", False):
                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        input_y = input[0, :, -1].reshape(-1, 1)
                        input_original = test_data.inverse_transform(input_y)
                        input_original = input_original.reshape(-1)

                        gt = np.concatenate((input_original, true_original[0, :, -1]), axis=0)
                        pd = np.concatenate((input_original, pred_original[0, :, -1]), axis=0)

                        plt.figure(figsize=(10, 5))
                        plt.plot(gt, label="Ground Truth", color="blue")
                        plt.plot(pd, label="Prediction", color="red")
                        plt.legend()
                        plt.title("Prediction vs Ground Truth")

                        img_path = os.path.join(folder_path, "final_batch_visualization.png")
                        plt.savefig(img_path)
                        plt.close()

                        wandb.log({"Final Batch Visualization": wandb.Image(img_path)})

        preds = np.array(preds)
        trues = np.array(trues)

        # ✅ 결과 저장 (원본 코드 유지)
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, r2, adj_r2, d_stat = metric(preds, trues, self.config["enc_in"])

        if self.final_run and self.config.get("use_wandb", False):
            print("self.final_run",self.final_run)
            wandb.log({
                "test_mae": mae,
                "test_mape": mape,
                "test_r2": r2,
                "test_adj_r2": adj_r2,
                "test_d_stat": d_stat
            }, step=999999)

        print('mape:{:.4f}, mae:{:.2f}, adj_r2:{:.2f}, r2:{:.2f}, d_stat:{:.2f}'.format(mape, mae, adj_r2, r2, d_stat))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mape:{}, mae:{}'.format(mape, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return