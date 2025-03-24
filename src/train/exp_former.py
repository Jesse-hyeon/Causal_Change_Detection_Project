import sys
import os

import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import warnings

import wandb

from src.utils.etc import EarlyStopping, adjust_learning_rate, metric

### data
from src.train.data_custom_former import data_provider

### Model
from src.model.NST import ns_transformer_model
from src.model.Transformer import transformer_model
from src.model.NSI import ns_informer_model
from src.model.NSA import ns_autoformer_model

class Exp_Basic_former(object):
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

### 2) Main
class Exp_Main_former(Exp_Basic_former):
    def __init__(self, config, final_run):
        super(Exp_Main_former, self).__init__(config)
        self.final_run = final_run

    def _build_model(self):
        model_dict = {
            'ns_transformer': ns_transformer_model,
            'transformer': transformer_model,
            'ns_informer': ns_informer_model,
            'ns_autoformer': ns_autoformer_model
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.config["pred_len"]:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.config["label_len"], :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.config["use_amp"]:
                    with torch.cuda.amp.autocast():
                        if self.config["output_attention"]:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.config["output_attention"]:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.config["features"] == 'MS' else 0
                outputs = outputs[:, -self.config["pred_len"]:, f_dim:]
                batch_y = batch_y[:, -self.config["pred_len"]:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')

        print("start training")

        # ✅ final_run이 아닌 경우에만 validation 데이터 로드 및 early stopping 준비
        if not self.final_run:
            vali_data, vali_loader = self._get_data(flag='val')
            path = os.path.join(self.config["checkpoints"], setting)
            if not os.path.exists(path):
                os.makedirs(path)
            early_stopping = EarlyStopping(patience=self.config["patience"], verbose=True)

        time_now = time.time()

        train_steps = len(train_loader)  # batch 개수를 저장
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):  # 하나의 Batch씩 데이터를 불러옴
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)  # batch_x -> batch 안에 있는 각 sample의 입력값(sequence_length)

                batch_y = batch_y.float().to(self.device)  # batch_y -> batch 안에 있는 pred의 실제값(pred_length)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.config["pred_len"]:, :]).float()  # batch_y에 대한 예측
                dec_inp = torch.cat([batch_y[:, :self.config["label_len"], :], dec_inp], dim=1).float().to(
                    self.device)  # label_len 만큼 데이터를 앞에 붙여 사용

                # encoder - decoder
                if self.config["use_amp"]:
                    with torch.cuda.amp.autocast():
                        if self.config["output_attention"]:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.config["features"] == 'MS' else 0  ### output은 시점이 겹치는 경우 어떻게 처리하는가? -> 최신 예측값 사용용
                        outputs = outputs[:, -self.config["pred_len"]:, f_dim:]
                        batch_y = batch_y[:, -self.config["pred_len"]:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.config["output_attention"]:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

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

                # 역전파 및 가중치 업데이트
                if self.config["use_amp"]:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # 조기 종료 및 학습률 조정
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            if not self.final_run:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)
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
        print("start testing")
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.config["pred_len"]:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.config["label_len"], :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.config["use_amp"]:
                    with torch.cuda.amp.autocast():
                        if self.config["output_attention"]:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.config["output_attention"]:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.config["features"] == 'MS' else 0
                outputs = outputs[:, -self.config["pred_len"]:, f_dim:]
                batch_y = batch_y[:, -self.config["pred_len"]:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                # pred와 true의 shape이 (1, 90, 1) 즉, (batch_size=1, seq_len=90, feature_dim=1)
                batch_size, seq_len, feature_dim = pred.shape  # (1, 90, 1)

                # 3D -> 2D 변환 (reshape)
                pred_reshaped = pred.reshape(seq_len, feature_dim)  # (90, 1)
                true_reshaped = true.reshape(seq_len, feature_dim)  # (90, 1)

                # 역변환 (inverse transform)
                pred_original = test_data.inverse_transform(pred_reshaped)  # (90, 1)
                true_original = test_data.inverse_transform(true_reshaped)  # (90, 1)

                # 다시 3D로 변환 (원래 차원 복원)
                pred_original = pred_original.reshape(batch_size, seq_len, feature_dim)  # (1, 90, 1)
                true_original = true_original.reshape(batch_size, seq_len, feature_dim)  # (1, 90, 1)

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
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
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