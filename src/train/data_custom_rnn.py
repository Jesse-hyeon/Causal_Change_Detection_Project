import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='b', feature_set=None):
        """
        seq_len: LSTM에 들어갈 입력 길이
        pred_len: 예측해야 할 길이
        """
        # 기본 시퀀스 길이 설정
        if size is None:
            self.seq_len = 96
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]  # ✅ label_len 제거 (LSTM에서는 불필요)

        # 데이터 관련 설정
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.feature_set = feature_set

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.scaler_y = StandardScaler()  # ✅ y값을 위한 별도 스케일러
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # 선택된 feature만 사용
        df_raw = df_raw[self.feature_set]

        # 날짜 컬럼 정리
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        # 데이터셋 분할 (80% train, 10% val, 10% test)
        num_train = int(len(df_raw) * 0.9)
        num_test = int(len(df_raw) * 0.05)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw[cols] # ✅ 전체 feature에 대한 데이터
        df_target = df_raw[[self.target]]  # ✅ y 값만 있는 데이터

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            # train_data = df_data[border1s[0]:border2s[0]]
            # self.scaler.fit(train_data.values)
            # data = self.scaler.transform(df_data.values)

            train_data = df_data[border1s[0]:border2s[0]]
            train_target = df_target[border1s[0]:border2s[0]]

            self.scaler.fit(train_data.values)  # ✅ 전체 feature scaling
            self.scaler_y.fit(train_target.values)  # ✅ y 값만 scaling

            data = self.scaler.transform(df_data.values)
            data_y = self.scaler_y.transform(df_target.values)  # ✅ y 값만 따로 scaling
        else:
            data = df_data.values

        # df_stamp = df_raw[['date']][border1:border2]
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values

        elif self.timeenc == 1:
            None

        self.data_stamp = data_stamp

        ############ 미치겠네 이게 맞나 싶음 ############
        y = data[:, -1].reshape(-1, 1)
        x = data[:, :-1]

        data = np.concatenate((x, data_stamp, y), axis=1)
        ############ 미치겠네 이게 맞나 싶음 ############

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        """
        LSTM 모델을 위해 `seq_len`만큼의 x 입력과
        `pred_len` 만큼의 y(target) 반환
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]  # LSTM 예측 대상 데이터 (pred_len)

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler_y.inverse_transform(data)

data_dict = {
    'custom': Dataset_Custom,
}

def data_provider(config, flag):
    """
    LSTM 모델에 맞는 데이터 로더를 생성하는 함수
    """
    Data = data_dict[config["data"]]
    timeenc = 0 if config["embed"] != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1  # 테스트는 batch_size=1
        freq = config["freq"]
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = config["batch_size"]
        freq = config["freq"]

    data_set = Data(
        root_path=config["root_path"],
        data_path=config["data_path"],
        flag=flag,
        size=[config["seq_len"], config["pred_len"]],
        features=config["features"],
        target=config["target"],
        timeenc=timeenc,
        freq=freq,
        feature_set = config["feature_set"]
    )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=config["num_workers"],
        drop_last=drop_last)

    return data_set, data_loader