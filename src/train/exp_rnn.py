import sys
import os
sys.path.append(os.path.abspath("src"))

import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from utils.etc import EarlyStopping, adjust_learning_rate, metric
import wandb

### data
from train.data_custom_rnn import data_provider

### Model
from model.LSTM import lstm_model
from model.MLP import mlp_model

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
            'mlp': mlp_model
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
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # ✅ LSTM에서는 인코더-디코더 구조가 없으므로 바로 예측
                if self.config["use_amp"]:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                # ✅ 원래 코드처럼 예측값과 정답을 마지막 `pred_len` 구간에서 비교
                f_dim = -1 if self.config["features"] == 'MS' else 0
                outputs = outputs[:, -self.config["pred_len"]:, f_dim:]
                batch_y = batch_y[:, -self.config["pred_len"]:, f_dim:]

                # ✅ 원래 코드처럼 detach() 후 loss 계산
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.config["checkpoints"], setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.config["patience"], verbose=True)

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

                # ✅ 예측 수행
                if self.config["use_amp"]:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                # ✅ feature type에 따른 차원 선택
                f_dim = -1 if self.config["features"] == 'MS' else 0

                outputs = outputs[:, -self.config["pred_len"]:, f_dim:]
                batch_y = batch_y[:, -self.config["pred_len"]:, f_dim:].to(self.device)


                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.config["train_epochs"] - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # ✅ 역전파 및 가중치 업데이트 (순서 유지)
                if self.config["use_amp"]:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # ✅ 원래 코드와 동일한 순서 유지
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.config)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

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

                if self.final_run:
                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        input_y = input[0, :, -1].reshape(-1, 1)
                        input_original = test_data.inverse_transform(input_y)
                        input_original = input_original.reshape(-1)

                        gt = np.concatenate((input_original, true_original[0, :, -1]), axis=0)
                        pd = np.concatenate((input_original, pred_original[0, :, -1]), axis=0)

                        # Matplotlib으로 시각화
                        plt.figure(figsize=(10, 5))
                        plt.plot(gt, label="Ground Truth", color="blue")
                        plt.plot(pd, label="Prediction", color="red")
                        plt.legend()
                        plt.title("Prediction vs Ground Truth")

                        # 파일 저장
                        img_path = os.path.join(folder_path, "final_batch_visualization.png")
                        plt.savefig(img_path)
                        plt.close()

                        # ✅ WandB에 이미지 업로드
                        wandb.log({"Final Batch Visualization": wandb.Image(img_path)})

        preds = np.array(preds)
        trues = np.array(trues)

        print("preds shape: ", preds.shape)
        print("trues shape: ", trues.shape)

        # ✅ 결과 저장 (원본 코드 유지)
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, d_stat = metric(preds, trues)

        if self.final_run:
            wandb.log({
                "test_mae": mae,
                "test_mse": mse,
                "test_rmse": rmse,
                "test_mape": mape,
                "test_mspe": mspe,
                "test_d_stat": d_stat
            })

        print('mape:{:.4f}, mae:{:.2f}, d_stat:{:.2f}'.format(mape, mae, d_stat))
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