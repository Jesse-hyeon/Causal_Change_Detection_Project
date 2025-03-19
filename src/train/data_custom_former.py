import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='b', feature_set=None,
                 num_train = 0, num_valid = 0):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.feature_set = feature_set

        self.num_train = num_train
        self.num_valid = num_valid

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.scaler_y = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        ### 변수 선택
        df_raw = df_raw[self.feature_set]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = self.num_train
        num_test = int(len(df_raw) * 0.1)
        num_vali = self.num_valid

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw[cols]
        df_target = df_raw[[self.target]]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            train_target = df_target[border1s[0]:border2s[0]]

            self.scaler.fit(train_data.values)
            self.scaler_y.fit(train_target.values)

            data = self.scaler.transform(df_data.values)
            data_y = self.scaler_y.transform(df_target.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values

        elif self.timeenc == 1:
            None

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler_y.inverse_transform(data)


### 2) Data Provider
data_dict = {
    'custom': Dataset_Custom,
}

def data_provider(config, flag):
    Data = data_dict[config["data"]]
    timeenc = 0 if config["embed"] != 'timeF' else 1

    if flag == 'train':
        shuffle_flag = False
        drop_last = True
        batch_size = config["batch_size"]
        freq = config["freq"]
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = config["freq"]

    data_set = Data(
        root_path=config["root_path"],
        data_path=config["data_path"],
        flag=flag,
        size=[config["seq_len"], config["label_len"], config["pred_len"]],
        features=config["features"],
        target=config["target"],
        timeenc=timeenc,
        freq=freq,
        feature_set=config["feature_set"],
        num_train = config["num_train"],
        num_valid = config["num_valid"]
    )

    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=config["num_workers"],
        drop_last=drop_last)

    return data_set, data_loader