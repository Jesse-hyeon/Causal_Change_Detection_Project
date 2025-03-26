import torch
import torch.nn as nn

class rnn_model(nn.Module):
    def __init__(self, config):
        super(rnn_model, self).__init__()

        self.num_classes = config["c_out"]
        self.num_layers = config["num_layers"]
        self.input_size = config["enc_in"]
        self.hidden_size = config["hidden_size"]
        self.seq_length = config["seq_len"]
        self.pred_len = config["pred_len"]

        # 🔵 RNN 네트워크 정의
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True, nonlinearity='tanh')

        # 🔵 최종 출력 레이어
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # 🔵 초기 hidden state 정의 (RNN은 cell state 없음)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # 🔵 RNN 실행
        rnn_out, _ = self.rnn(x, h_0)  # (batch_size, seq_len, hidden_size)

        # 🔵 마지막 pred_len만 선택
        rnn_out = rnn_out[:, -self.pred_len:, :]  # (batch_size, pred_len, hidden_size)

        # 🔵 예측값 계산
        output = self.fc(rnn_out)  # (batch_size, pred_len, num_classes)

        return output