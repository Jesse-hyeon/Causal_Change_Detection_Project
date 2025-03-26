import torch
import torch.nn as nn

class lstm_model(nn.Module):
    def __init__(self, config):
        super(lstm_model, self).__init__()

        self.num_classes = config["c_out"]
        self.num_layers = config["num_layers"]
        self.input_size = config["enc_in"]
        self.hidden_size = config["hidden_size"]
        self.seq_length = config["seq_len"]
        self.pred_len = config["pred_len"]

        # LSTM 네트워크 정의
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)

        # 최종 예측을 위한 Linear 레이어
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # 🟢 초기 hidden state 및 cell state 정의
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # 🟢 LSTM 실행
        lstm_out, _ = self.lstm(x, (h_0, c_0))  # (batch_size, seq_len, hidden_size)

        # 🟢 LSTM 마지막 `pred_len` 타임스텝만 선택
        lstm_out = lstm_out[:, -self.pred_len:, :]  # (batch_size, pred_len, hidden_size)

        # 🟢 최종 예측값 계산
        output = self.fc(lstm_out)  # (batch_size, pred_len, num_classes)

        return output


# import torch
# import torch.nn as nn
#
# class lstm_model(nn.Module):
#     def __init__(self, config):
#         super(lstm_model, self).__init__()
#
#         self.n_feature = config["c_out"]         # 예측하고자 하는 feature 수
#         self.hidden_size = config["hidden_size"]
#         self.pred_len = config["pred_len"]
#
#         self.lstm = nn.LSTM(
#             input_size=config["enc_in"],
#             hidden_size=self.hidden_size,
#             num_layers=config["num_layers"],
#             batch_first=True
#         )
#
#         self.fc = nn.Sequential(
#             nn.Linear(self.hidden_size, self.pred_len * self.n_feature),
#             nn.ReLU()  # 또는 필요 시 tanh, sigmoid 등
#         )
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         h_0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
#         c_0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
#
#         _, (h_n, _) = self.lstm(x, (h_0, c_0))   # h_n: (1, batch, hidden_size)
#         h_n = h_n.squeeze(0)                    # -> (batch, hidden_size)
#
#         out = self.fc(h_n)                      # -> (batch, pred_len * n_feature)
#         output = out.view(batch_size, self.pred_len, self.n_feature)
#
#         return output





