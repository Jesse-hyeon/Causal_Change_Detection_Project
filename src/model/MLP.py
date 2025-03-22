"""
Multilayer Perceptron (MLP)
"""
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class mlp_model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.configs = config
        self.pred_len = config["pred_len"]

        # ✅ 입력 차원: seq_len × enc_in
        input_dim = config["seq_len"] * config["enc_in"]
        hidden_dims = config["p_hidden_dims"]

        # ✅ 히든 레이어 구성
        hidden_params = [(input_dim, hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            hidden_params.append((hidden_dims[i - 1], hidden_dims[i]))

        layers = OrderedDict()
        for idx, params in enumerate(hidden_params):
            layers[f'linear_{idx + 1}'] = nn.Linear(*params, bias=True)
            layers[f'sigmoid_{idx + 1}'] = nn.Sigmoid()
        self.layers = nn.Sequential(layers)

        # ✅ 출력층
        last_dim = hidden_params[-1][-1]
        self.output = nn.Linear(last_dim, self.pred_len)

        # ✅ weight 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(1.0)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = x.view(x.shape[0], -1)  # flatten
        h = self.layers(x)
        out = self.output(h)  # (batch_size, pred_len)
        return out.unsqueeze(-1)   # (batch_size, pred_len, 1)

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.configs["learning_rate"],
            momentum=self.configs.get("momentum", 0.9)  # 기본값 0.9
        )

    def get_regularization(self, regularizer='L2', strength=0.1):
        reg = torch.tensor(0.0, device=self.device)
        for params in self.parameters():
            if regularizer == 'L1':
                reg += torch.norm(params, 1)
            elif regularizer == 'L2':
                reg += torch.norm(params, 2)
        return reg * strength

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, :, -1]  # 타깃만 선택
        preds = self(x).squeeze()
        loss = F.mse_loss(preds, y)

        if self.configs.get("regularizer", None):
            loss += self.get_regularization(
                self.configs["regularizer"],
                self.configs.get("lambda", 0.1)
            )

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, :, -1]
        preds = self(x).squeeze()
        loss = F.mse_loss(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, :, -1]
        preds = self(x).squeeze()
        loss = F.mse_loss(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def predict(self, x):
        return self(x).detach().numpy()

    def get_weights(self, to_numpy=True):
        weights = torch.Tensor()
        for name, param in self.named_parameters():
            if "weight" in name:
                weights = torch.cat((weights, param.view(-1)), dim=0)
        return weights.detach().numpy() if to_numpy else weights
