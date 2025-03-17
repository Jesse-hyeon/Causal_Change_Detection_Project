import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.num_classes = configs.c_out
        self.num_layers = configs.num_layers
        self.input_size = configs.input_size
        self.hidden_size = configs.hidden_size
        self.seq_length = configs.seq_len
        self.pred_len = configs.pred_len

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