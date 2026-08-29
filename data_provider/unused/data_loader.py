import os
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import pywt
from scipy.stats import median_abs_deviation
from scipy.stats import spearmanr
from scipy.signal import detrend

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=10, 
                 task_name='long_term_forecast', is_pretraining=1):
        # size [seq_len, label_len, pred_len]
        # info
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

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # if self.set_type == 0:
        #     border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 
            

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        

    def __getitem__(self, index):
        s_begin = index
        if self.set_type == 0 & self.percent != 100:
            s_begin = int(s_begin * (1 // self.percent))
            
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0 & self.percent != 100:
            return int((len(self.data_x) - self.seq_len - self.pred_len + 1) * self.percent)
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='M', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=10,
                 task_name='long_term_forecast', is_pretraining=1):
        # size [seq_len, label_len, pred_len]
        # info
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
        self.flag = flag

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # if self.set_type == 0:
        #     border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        

    def __getitem__(self, index):
        s_begin = index
        
        if self.set_type == 0 & self.percent != 100:
            s_begin = int(s_begin * (1 // self.percent))
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0 & self.percent != 100:
            return int((len(self.data_x) - self.seq_len - self.pred_len + 1) * self.percent)
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=10,
                 task_name='long_term_forecast', is_pretraining=1):
        # size [seq_len, label_len, pred_len]
        # info
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

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]
        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]   
        
        # if self.set_type == 0:
        #     border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len       
            
        
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        

    def __getitem__(self, index):
        s_begin = index
        
        if self.set_type == 0 & self.percent != 100:
            s_begin = int(s_begin * (1 // self.percent))
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0 & self.percent != 100:
            return int((len(self.data_x) - self.seq_len - self.pred_len + 1) * self.percent)
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=10, 
                 task_name='long_term_forecast', is_pretraining=1):
        # size [seq_len, label_len, pred_len]
        # info
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

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        
        # if self.set_type == 0:
        #     s_begin = int(s_begin * (1 // self.percent))
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 4))
        seq_y_mark = torch.zeros((seq_x.shape[0], 4))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # if self.set_type == 0:
        #     return int((len(self.data_x) - self.seq_len - self.pred_len + 1) * self.percent)
        # else:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    


class Dataset_IceCore_M(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='icecore_processed.npz', 
                 features='M', target='OT', scale=True, timeenc=0, freq='h', percent=100, 
                 task_name='long_term_forecast', is_pretraining=0):
        
        # 补全初始化参数，与 data_factory 传入的参数对齐
        if size == None:
            self.seq_len = 64
            self.label_len = 32
            self.pred_len = 16
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # 注意：冰芯数据由于年份跨度长，滑动步长可以根据需要调节
        self.stride = 2 if flag == 'train' else 1
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.root_path = root_path
        self.data_path = data_path

        self.scale = False
        
        self.__read_data__()

    def __read_data__(self):
        file_path = os.path.join(self.root_path, self.data_path)
        npz_data = np.load(file_path, allow_pickle=True)
        
        self.all_dynamic_data = []
        self.all_static_data = []
        self.window_indices = []
        
        # 遍历所有站点
        for site_name in npz_data.files:
            site_dict = npz_data[site_name].item()
            dynamic_feat = site_dict['dynamic'][::-1].copy()  # [Time, 3]
            static_feat = site_dict['static']    # [4]
            
            # --- 核心新增：时间序列按 70% : 10% : 20% 划分 ---
            total_len = len(dynamic_feat)
            train_len = int(total_len * 0.7)
            val_len = int(total_len * 0.1)
            
            if self.set_type == 0:   # train
                dynamic_feat = dynamic_feat[:train_len]
            elif self.set_type == 1: # val
                dynamic_feat = dynamic_feat[train_len : train_len + val_len]
            else:                    # test
                dynamic_feat = dynamic_feat[train_len + val_len :]
            
            # --- 核心新增：确保剩余长度足够切出一个完整的 (seq_len + pred_len) 样本 ---
            n_timepoint = (len(dynamic_feat) - self.seq_len - self.pred_len) // self.stride + 1
            
            if n_timepoint <= 0:
                continue
                
            start_idx = len(self.all_dynamic_data)
            self.all_dynamic_data.append(dynamic_feat)
            self.all_static_data.append(static_feat)
            
            # 记录有效的滑窗起始点
            for w in range(n_timepoint):
                s_begin = w * self.stride
                self.window_indices.append((start_idx, s_begin))

    def __getitem__(self, index):
        site_idx, s_begin = self.window_indices[index]
        
        # --- 核心新增：构建 Encoder 输入 (x) 和 Decoder 输入/标签 (y) ---
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # Encoder 过去输入
        seq_x = self.all_dynamic_data[site_idx][s_begin:s_end, :] 
        # Decoder 监督标签
        seq_y = self.all_dynamic_data[site_idx][r_begin:r_end, :] 
        
        # 提取当前站点的静态特征 (经纬度海拔)
        static_x = self.all_static_data[site_idx]                 
        
        # 沿着时间维度 (第0轴) 复制，强行对齐动态序列的时间长度
        # seq_x_mark 形状变为: [seq_len, 4]
        seq_x_mark = np.tile(static_x, (self.seq_len, 1)) 
        
        # seq_y_mark 形状变为: [label_len + pred_len, 4]
        seq_y_mark = np.tile(static_x, (self.label_len + self.pred_len, 1))        
        # 返回标准的四元组格式，将 static_x 放在 mark 的位置返回，避免解包报错
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.window_indices)
    
class Dataset_IceCore_Accum_Year(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='antarctic_aligned.csv',
                 target='snow', scale=True, timeenc=0, freq='y',
                 percent=100, task_name='long_term_forecast', is_pretraining=0): # 1. 补全框架必传参数

        # 2. 增加 size=None 的默认容错
        if size == None:
            self.seq_len = 64
            self.label_len = 32
            self.pred_len = 16
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.percent = percent  # 记录少样本比例
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = df_raw.sort_values(by='year', ascending=True).reset_index(drop=True)
        # 排除 date 列，剩下的每一列都当作一个独立的冰芯站点（单变量序列）
        sites = [col for col in df_raw.columns if col != 'year']
        
        self.data_x_list = []
        self.data_y_list = []
        self.data_stamp_list = []
        self.valid_indices = []
        
        train_data_all = []
        site_data_dict = {}
        
        for site in sites:
            # 单独提取该站点数据并去除缺失值，保证序列连续性
            df_site = df_raw[['year', site]].dropna()
            length = len(df_site)
            
            # 若该站点总长度不足以构成一个样本，则跳过
            if length < self.seq_len + self.pred_len:
                continue
                
            num_train = int(length * 0.6)
            num_test = int(length * 0.2)
            num_vali = length - num_train - num_test
            
            # 仿照 ETTh 逻辑，设置每个站点内部的切分边界（利用 seq_len 形成 overlap 保证特征输入完整）
            border1s = [0, num_train - self.seq_len, length - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, length]
            
            site_data_dict[site] = {
                'df': df_site,
                'border1s': border1s,
                'border2s': border2s
            }
            
            # 收集所有站点的训练集数据，用于拟合全局 Scaler
            train_data = df_site[[site]].values[border1s[0]:border2s[0]]
            train_data_all.append(train_data)
            
        # 在所有站点的训练数据上 Fit 全局 Scaler
        if self.scale and len(train_data_all) > 0:
            train_data_all = np.vstack(train_data_all)
            self.scaler.fit(train_data_all)
            
        site_idx = 0
        for site, info in site_data_dict.items():
            df_site = info['df']
            border1 = info['border1s'][self.set_type]
            border2 = info['border2s'][self.set_type]
            
            df_data = df_site[[site]].values
            if self.scale:
                data = self.scaler.transform(df_data)
            else:
                data = df_data
                
            df_stamp = df_site[['year']][border1:border2]
            
            # 兼容处理年份（如 1800）和标准日期格式
            try:
                df_stamp['year'] = pd.to_datetime(df_stamp['year'].astype(str), format='%Y')
            except ValueError:
                df_stamp['year'] = pd.to_datetime(df_stamp['year'])
                
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
                data_stamp = df_stamp.drop(['year'], axis=1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['year'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
                
            data_x = data[border1:border2]
            data_y = data[border1:border2]
            
            self.data_x_list.append(data_x)
            self.data_y_list.append(data_y)
            self.data_stamp_list.append(data_stamp)
            
            # 计算当前站点在当前 flag 下可以滑动的窗口数
            seq_count = len(data_x) - self.seq_len - self.pred_len + 1
            for i in range(max(0, seq_count)):
                self.valid_indices.append((site_idx, i))
                
            site_idx += 1
            
        # 3. 落实 Few-shot 截断逻辑：按比例随机/顺序舍弃训练集索引
        if self.set_type == 0 and self.percent < 100:
            keep_len = int(len(self.valid_indices) * (self.percent / 100))
            self.valid_indices = self.valid_indices[:keep_len]

    def __getitem__(self, index):
        site_idx, s_begin = self.valid_indices[index]
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x_list[site_idx][s_begin:s_end]
        seq_y = self.data_y_list[site_idx][r_begin:r_end]
        seq_x_mark = self.data_stamp_list[site_idx][s_begin:s_end]
        seq_y_mark = self.data_stamp_list[site_idx][r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.valid_indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_IceCore_Accum_Site(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='antarctic_aligned.csv',
                 target='snow', scale=True, timeenc=0, freq='y',
                 percent=100, task_name='long_term_forecast', is_pretraining=0,
                 mode='random', apply_pywt=False, site_meta_path='dataset/icecore/data_sources.csv',
                 use_spectral_features=1, cluster_result_path='cluster_results_cosine_zscore.csv',
                 residual_prediction=False, pred_seperate=None,
                 support_ratio=None):

        # self.use_spectral_features = use_spectral_features
        self.cluster_result_path = cluster_result_path
        self.residual_prediction = residual_prediction
        self.support_ratio = support_ratio  # None = 固定100年; 0.66 = 按比例
        # import pdb;pdb.set_trace()
        self.pred_seperate = pred_seperate
        self.pred_seperate_list = None
        if self.pred_seperate is not None:
            if isinstance(self.pred_seperate, str):
                self.pred_seperate_list = [int(x) for x in self.pred_seperate.strip().split()]
            elif isinstance(self.pred_seperate, list):
                self.pred_seperate_list = [int(x) for x in self.pred_seperate]
            else:
                self.pred_seperate_list = [int(self.pred_seperate)]

        if size == None:
            self.seq_len = 64
            self.label_len = 32
            self.pred_len = 16
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.site_meta_path = site_meta_path
        self.task_name = task_name
        self.is_pretraining = is_pretraining
        
        assert mode in ['random', 'region', 'cluster'], "mode 必须为 'random'、'region' 或 'cluster'"
        self.mode = mode
        self.apply_pywt = apply_pywt
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.static_scaler = StandardScaler()
        # self.scalers = {}

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = df_raw.sort_values(by='year', ascending=True).reset_index(drop=True)

       
        cluster_df_filter = pd.read_csv(self.cluster_result_path)
        cluster_df_filter.iloc[:, 0] = (cluster_df_filter.iloc[:, 0]
                                        .astype(str).str.strip().str.replace('"', '', regex=False))
        cluster_map_filter = cluster_df_filter.set_index(cluster_df_filter.columns[0])['Level_0'].to_dict()
        
        
        # cols_to_drop = []
        # for col in df_raw.columns:
        #     if col == 'year':
        #         continue
        #     clean_col = col.strip().replace('"', '')
        #     ori_cluster = cluster_map_filter.get(clean_col, -1)
        #     merged_cluster = merge_map.get(ori_cluster, ori_cluster)
            
        #     if self.pred_seperate_list is not None:
        #         # 如果设置了开关，仅保留位于 pred_seperate_list 的合并后簇，其余全部剔除
        #         if merged_cluster not in self.pred_seperate_list:
        #             cols_to_drop.append(col)
        #     # else:
        #     #     # 原有的手动硬编码降级逻辑 (如果没有使用开关)
        #     #     if ori_cluster in [1, 2, 3, 4, 6]:
        #     #         cols_to_drop.append(col)
                
        # # 从 DataFrame 中物理删除这些列
        # if cols_to_drop:
        #     df_raw = df_raw.drop(columns=cols_to_drop)
        #     if self.pred_seperate_list is not None:
        #         print(f"[{self.mode} mode] 已根据 pred_seperate={self.pred_seperate_list} 提前剔除 {len(cols_to_drop)} 个站点。")
        #     else:
        #         print(f"[{self.mode} mode] 已提前剔除 {len(cols_to_drop)} 个属于合并前 Cluster 1/2/3 的站点。")
        
        df_meta = pd.read_csv(self.site_meta_path)
        df_meta['site_name_clean'] = df_meta['site'].astype(str).str.strip().str.replace('"', '', regex=False)
        site_meta_dict = df_meta.set_index('site_name_clean')[['Latitude', 'Longitude', 'Elevation']].to_dict('index')

        all_sites = [col for col in df_raw.columns if col != 'year']

        df_copy = df_raw.copy()
        if self.apply_pywt == True:
            print("="*60)
            print("Apply pywt...")
            for site in all_sites:
                series = df_copy[site]
                valid_idx = series.dropna().index
                valid_data = series.dropna().values
                N = len(valid_data)
                
                if N == 0:
                    continue
                    
                # 1. 解决 SWT 的 Padding 问题 (level=1, 长度必须是 2 的倍数)
                pad_width = 1 if N % 2 != 0 else 0
                if pad_width > 0:
                    x = np.pad(valid_data, (0, pad_width), 'symmetric')
                else:
                    x = valid_data
                    
                # 保护机制
                if pywt.swt_max_level(len(x)) < 1:
                    continue
                    
                # 2. 1层小波变换
                coeffs = pywt.swt(x, 'db2', level=1, start_level=0)
                cA1, cD1 = coeffs[0]
                
                # 3. 稳健噪音估计与软阈值去噪
                sigma = median_abs_deviation(cD1, scale='normal')
                if sigma > 0:
                    threshold = sigma * np.sqrt(2 * np.log(N))
                    cD1_clean = pywt.threshold(cD1, value=threshold, mode='soft')
                else:
                    cD1_clean = cD1
                    
                # 4. 逆变换重构并移除 padding
                clean_data = pywt.iswt([(cA1, cD1_clean)], 'db2')
                if pad_width > 0:
                    clean_data = clean_data[:-pad_width]
                    
                df_copy.loc[valid_idx, site] = clean_data
            print("pywt done!")
            print("="*60)
                
        valid_sites = []
        invalid_sites = []
        
        min_len_threshold = self.seq_len + self.pred_len
        
        for site in all_sites:
            df_site = df_copy[['year', site]].dropna()
            length = len(df_site)
            
            if length < min_len_threshold:
                invalid_sites.append(f"{site}({length}年)")
            else:
                valid_sites.append(site)
                
        train_sites = []
        val_sites = []
        test_sites = []
        
        # 设置随机种子，保证每次实例化切分结果绝对一致
        np.random.seed(2030)
        
    
        if self.mode == 'cluster':
            print("Dividing train/val/test per cluster...")
            
            # 读取聚类结果，构建站点->簇的映射
            cluster_df = pd.read_csv(self.cluster_result_path)
            cluster_df.iloc[:, 0] = (cluster_df.iloc[:, 0]
                          .astype(str).str.strip().str.replace('"', '', regex=False))
            cluster_map = cluster_df.set_index(cluster_df.columns[0])['Level_0'].to_dict()
            cluster_dict = {}
            for s in valid_sites:
                clean_s = s.strip().replace('"', '')
                cid = cluster_map.get(clean_s, -1)  # -1 = 未匹配
                cluster_dict.setdefault(cid, []).append(s)
            
            # 各簇内部独立切分，逻辑和region完全一致
            for cid, c_sites in cluster_dict.items():
                np.random.shuffle(c_sites)
                n = len(c_sites)
                if n == 1:
                    train_sites.extend(c_sites)
                elif n == 2:
                    train_sites.append(c_sites[0])
                    test_sites.append(c_sites[1])
                else:
                    r_train_cnt = int(n * 0.6)
                    r_remainder = n - r_train_cnt
                    r_val_cnt = r_remainder // 2
                    r_test_cnt = r_remainder - r_val_cnt
                    train_sites.extend(c_sites[:r_train_cnt])
                    val_sites.extend(c_sites[r_train_cnt:r_train_cnt + r_val_cnt])
                    test_sites.extend(c_sites[r_train_cnt + r_val_cnt:])

        # if cols_to_drop:
        #     train_sites = [s for s in train_sites if s not in cols_to_drop]
        #     val_sites   = [s for s in val_sites   if s not in cols_to_drop]
        #     test_sites  = [s for s in test_sites  if s not in cols_to_drop]

        # 仅在构建 train 数据集时打印一次划分日志
        if self.set_type == 0:
            print("="*60)
            print(f"数据划分模式: {self.mode}")
            print(f"剔除的站点 (不足 {self.seq_len + self.pred_len} 年): {invalid_sites}")
            print("-" * 60)
            print(f"Train 站点分配 ({len(train_sites)}个): {train_sites}")
            print(f"Val 站点分配 ({len(val_sites)}个): {val_sites}")
            print(f"Test 站点分配 ({len(test_sites)}个): {test_sites}")

            # ===== 新增：各簇在 train/val/test 中的分布 =====
            cluster_df_log = pd.read_csv(self.cluster_result_path)
            cluster_df_log.iloc[:, 0] = (cluster_df_log.iloc[:, 0]
                                        .astype(str).str.strip()
                                        .str.replace('"', '', regex=False))
            cluster_map_log = cluster_df_log.set_index(
                cluster_df_log.columns[0])['Level_0'].to_dict()

            def count_by_cluster(sites):
                counter = {}
                for s in sites:
                    cid = cluster_map_log.get(s.strip().replace('"', ''), -1)
                    counter[cid] = counter.get(cid, 0) + 1
                return counter

            train_cnt = count_by_cluster(train_sites)
            val_cnt   = count_by_cluster(val_sites)
            test_cnt  = count_by_cluster(test_sites)
            all_cids  = sorted(set(train_cnt) | set(val_cnt) | set(test_cnt))

            print(f"{'簇ID':<8} {'Train':>6} {'Val':>6} {'Test':>6} {'合计':>6}")
            print("-" * 36)
            for cid in all_cids:
                t = train_cnt.get(cid, 0)
                v = val_cnt.get(cid, 0)
                s = test_cnt.get(cid, 0)
                label = f"Cluster {cid}" if cid != -1 else "未匹配"
                print(f"{label:<8} {t:>6} {v:>6} {s:>6} {t+v+s:>6}")
            print("-" * 36)
            print(f"{'合计':<8} {len(train_sites):>6} {len(val_sites):>6} "
            f"{len(test_sites):>6} {len(train_sites)+len(val_sites)+len(test_sites):>6}")
            # ===== 新增结束 =====
            print("="*60)

        
        # import pdb;pdb.set_trace()
        # 根据当前的 flag 选择对应的站点集合
        if self.set_type == 0:
            self.current_sites = train_sites
        elif self.set_type == 1:
            self.current_sites = val_sites
        else:
            self.current_sites = test_sites
        current_sites = self.current_sites

        # 3. 拟合 Scaler (仅使用 train_sites 的数据)
        if self.scale:
            train_data_all = []
            for site in train_sites:
                df_site = df_copy[['year', site]].dropna()
                train_data_all.append(df_site[[site]].values)
            if len(train_data_all) > 0:
                self.scaler.fit(np.vstack(train_data_all))


        self.site_cluster_map = {}
        for site_idx, site in enumerate(current_sites):
            clean_s = site.strip().replace('"', '')
            self.site_cluster_map[site_idx] = cluster_map.get(clean_s, -1)


        self.static_feat_list = []
        self.data_x_list = []
        self.data_y_list = []
        self.data_stamp_list = []
        self.year_list = []
        self.valid_indices = []
        
        site_idx = 0

        train_static = []
        for s in train_sites:
            clean_s = s.strip().replace('"', '')
            meta = site_meta_dict.get(clean_s, {'Latitude': 0, 'Longitude': 0, 'Elevation': 0})
            lon_rad = np.radians(meta['Longitude'])
            train_static.append([meta['Latitude'], np.sin(lon_rad), np.cos(lon_rad), meta['Elevation']])
        if train_static:
            self.static_scaler.fit(np.array(train_static))
        
        # 4. 提取当前集合的数据
        for site in current_sites:
            clean_s = site.strip().replace('"', '')
            meta = site_meta_dict.get(clean_s, {'Latitude': 0, 'Longitude': 0, 'Elevation': 0})
            lon_rad = np.radians(meta['Longitude'])
            # 形状: [lat, sin_lon, cos_lon, elevation]
            static_raw = np.array([[meta['Latitude'], np.sin(lon_rad), np.cos(lon_rad), meta['Elevation']]])
            # 使用在前面可能已经 fit 过的 self.static_scaler
            static_scaled = self.static_scaler.transform(static_raw)[0]
            self.static_feat_list.append(static_scaled)

            df_site = df_copy[['year', site]].dropna()
            df_data = df_site[[site]].values

            site_std = float(np.std(df_data, ddof=0))
            if site_std < 1e-6:
                site_std = 1.0

                                
            if self.scale:
                data = self.scaler.transform(df_data)
            else:
                data = df_data

                
            df_stamp = df_site[['year']].copy()
            df_stamp['year'] = pd.to_numeric(df_stamp['year'])
            df_stamp['year_norm'] = (df_stamp['year'] - df_stamp['year'].mean()) / df_stamp['year'].std()
            data_stamp = df_stamp[['year_norm']].values
                
            self.data_x_list.append(data)
            self.data_y_list.append(data)
            self.data_stamp_list.append(data_stamp)
            self.year_list.append(df_stamp['year'].values)
            seq_count = len(data) - self.seq_len - self.pred_len + 1
            for i in range(max(0, seq_count)):
                self.valid_indices.append((site_idx, i))
                
            site_idx += 1
        # 5. 少样本 (Few-shot) 截断逻辑
        if self.set_type == 0 and self.percent < 100:
            keep_len = int(len(self.valid_indices) * (self.percent / 100))
            self.valid_indices = self.valid_indices[:keep_len]

        # ================== MAML: per-site support_len + 簇原型 ==================
        # support_ratio=None → 固定 100 年 (兼容 accum); 有值时按比例算
        self.support_len_list = []
        for site_idx_tmp in range(len(self.data_x_list)):
            site_total = len(self.data_x_list[site_idx_tmp])
            if self.support_ratio is not None:
                sl = int(site_total * self.support_ratio)
                # 保证 support 至少能切出 1 个完整样本
                sl = max(sl, self.seq_len + self.pred_len)
                # 保证 query 至少能切出 1 个完整样本
                sl = min(sl, site_total - self.seq_len - self.pred_len)
                sl = max(sl, self.seq_len + self.pred_len)  # 再次保护
            else:
                sl = 100  # 原始默认
            self.support_len_list.append(sl)
        # 全局 support_len 取最小，用于原型对齐
        self.support_len = min(self.support_len_list) if self.support_len_list else 100
        
        self.prototypes = {0: [], 1: [], 2: []}
        # 收集训练集各簇的前 support_len 年序列（取全局最小值对齐）
        for site in train_sites:
            clean_s = site.strip().replace('"', '')
            cid = cluster_map.get(clean_s, -1)
            if cid in self.prototypes:
                df_site_proto = df_copy[['year', site]].dropna()
                site_data = df_site_proto[site].values[:self.support_len]
                if len(site_data) == self.support_len:
                    self.prototypes[cid].append(site_data)
        
        # 计算平均原型序列
        for cid in self.prototypes:
            if len(self.prototypes[cid]) > 0:
                self.prototypes[cid] = np.mean(self.prototypes[cid], axis=0)
            else:
                self.prototypes[cid] = np.zeros(self.support_len)

        if self.residual_prediction:
            self.cluster_time_means = {}
            for cid in range(3):
                c_sites = [s for s in train_sites if cluster_map.get(s.strip().replace('"', ''), -1) == cid]
                # 【新增】保护机制：如果该簇被剔除导致无站点，直接跳过
                if len(c_sites) == 0:
                    continue
                # 取出属于该簇的所有训练集站点
                c_sites = [s for s in train_sites if cluster_map.get(s.strip().replace('"', ''), -1) == cid]
                # 沿用 df_copy (包含了完整的总年份序列，如150年)
                stack_data = np.stack([df_copy[s].values for s in c_sites], axis=0)
                mean_data = stack_data.mean(axis=0) # shape: (total_len,)
                if self.scale:
                    mean_data = self.scaler.transform(mean_data.reshape(-1, 1)).flatten()
                self.cluster_time_means[cid] = mean_data


    def __getitem__(self, index):
        # 1. 获取当前 Task 对应的站点索引和名称
        site_idx = index
        site_name = self.current_sites[site_idx]
        
        # per-site support length
        site_support_len = self.support_len_list[site_idx]
        
        # 获取该站点的完整数据
        data_x = self.data_x_list[site_idx]
        data_y = self.data_y_list[site_idx]
        data_stamp = self.data_stamp_list[site_idx]
        
        total_len = len(data_x)
        window_size = self.seq_len + self.pred_len

        cluster_id = self.site_cluster_map.get(site_idx, 0)
        if getattr(self, 'residual_prediction', False):
            c_mean_full = self.cluster_time_means.get(cluster_id, np.zeros(total_len)).reshape(-1, 1)
            data_x_model = data_x - c_mean_full
            data_y_model = data_y - c_mean_full
        else:
            c_mean_full = np.zeros(total_len)
            data_x_model = data_x
            data_y_model = data_y
        
        # ================== 微观切片：Support Set (前 site_support_len 年) ==================
        support_x, support_y, support_x_mark, support_y_mark = [], [], [], []
        support_y_mean = []
        self.stride = 1
        for i in range(0, site_support_len - window_size + 1, self.stride):
            s_begin = i
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            
            support_x.append(data_x_model[s_begin:s_end])
            support_y.append(data_y_model[r_begin:r_end])
            support_x_mark.append(data_stamp[s_begin:s_end])
            support_y_mark.append(data_stamp[r_begin:r_end])
            support_y_mean.append(c_mean_full[r_begin + self.label_len : r_end])
            
        # ================== 微观切片：Query Set (site_support_len 之后) ==================
        query_x, query_y, query_x_mark, query_y_mark = [], [], [], []
        query_years = []
        query_x_mean = []
        query_y_mean = []

        # Query 的预测目标必须完全落在 site_support_len 之后
        q_start_idx = site_support_len - self.seq_len
        for i in range(q_start_idx, total_len - window_size + 1, self.stride):
            s_begin = i
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            
            query_x.append(data_x_model[s_begin:s_end])
            query_y.append(data_y_model[r_begin:r_end])
            query_x_mark.append(data_stamp[s_begin:s_end])
            query_y_mark.append(data_stamp[r_begin:r_end])

            pred_start_year = self.year_list[site_idx][r_begin + self.label_len]
            query_years.append(pred_start_year)

            query_x_mean.append(c_mean_full[s_begin : s_end])
            query_y_mean.append(c_mean_full[r_begin + self.label_len : r_end])
            
        # ================== 动态构建条件特征 ==================
        # a) 3 维 One-hot
        cluster_id = self.site_cluster_map.get(site_idx, 0)
        one_hot = [1.0 if cluster_id == c else 0.0 for c in range(3)]
        
        # b) 3 维地理特征 (提取原本已标准化的前 3 维：lat, sin_lon, cos_lon)
        geo_feat = self.static_feat_list[site_idx][:3].tolist()
        
        # c) 3 维局部统计量 —— 用前 site_support_len 年的数据算
        support_raw = data_x[:site_support_len].flatten()
        local_mean = np.mean(support_raw)
        local_std = np.std(support_raw)
        
        fft_power = np.abs(np.fft.rfft(support_raw - local_mean)) ** 2
        split = max(1, len(fft_power) // 5)
        local_f0 = fft_power[:split].sum() / (fft_power[split:].sum() + 1e-8)
        
        # d) 3 维原型相似度 (皮尔逊相关系数)
        # 截取与全局 support_len 对齐的长度来算相关
        proto_raw = data_x[:self.support_len].flatten()
        if self.pred_seperate_list is None:
            sim_features = []
            for cid in range(3):
                proto = self.prototypes.get(cid, np.zeros(self.support_len))
                if len(proto_raw) < self.support_len:
                    sim = 0.0
                elif np.std(proto_raw) < 1e-6 or np.std(proto) < 1e-6:
                    sim = 0.0
                else:
                    sim = np.corrcoef(proto_raw, proto)[0, 1]
                sim_features.append(sim)
                
            maml_static_feat = np.array(one_hot + sim_features + geo_feat + [local_mean, local_std, local_f0], dtype=np.float32)
        else:
            maml_static_feat = np.array(one_hot + geo_feat + [local_mean, local_std, local_f0], dtype=np.float32)


        # 转换为 Tensor，便于后续 batch 堆叠
        # 注意：此处返回的都是一个 Task 下的集合，shape 为 [N_windows, seq_len, vars]
        return (
            torch.tensor(np.array(support_x), dtype=torch.float32), 
            torch.tensor(np.array(support_y), dtype=torch.float32),
            torch.tensor(np.array(support_x_mark), dtype=torch.float32), 
            torch.tensor(np.array(support_y_mark), dtype=torch.float32),
            torch.tensor(np.array(query_x), dtype=torch.float32), 
            torch.tensor(np.array(query_y), dtype=torch.float32),
            torch.tensor(np.array(query_x_mark), dtype=torch.float32), 
            torch.tensor(np.array(query_y_mark), dtype=torch.float32),
            torch.tensor(maml_static_feat, dtype=torch.float32),
            torch.tensor(np.array(query_years), dtype=torch.float32),
            # torch.tensor(np.array(support_y_mean), dtype=torch.float32), 
            torch.tensor(np.array(query_x_mean), dtype=torch.float32),
            torch.tensor(np.array(query_y_mean), dtype=torch.float32)
        )

    def __len__(self):
        # return len(self.valid_indices)
        return len(self.current_sites)

    def inverse_transform(self, data,site_name):
        # return self.scaler.inverse_transform(data)
        if self.scale and site_name in self.scalers:
            return self.scalers[site_name].inverse_transform(data)
        return data


class Dataset_IceCore_CrossVar(Dataset):
    """
    跨变量预测专用 DataLoader:
    - 输入: x_csv (如 accum), y_csv (如 chem)
    - 输出: support_x/query_x 来自源变量，support_y/query_y 来自目标变量
    - 时间严格对齐: 同几年预测同几年 (非滞后外推)
    - static_feat 末尾追加: 该站点历史 pearson(x, y)
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='S', 
                 data_path=None,
                 x_data_path='accum.csv',      # 【新增】源变量文件
                 y_data_path='merged_chem_clean.csv',  # 【新增】目标变量文件
                 target='na', scale=True, timeenc=0, freq='y',
                 percent=100, task_name='long_term_forecast', is_pretraining=0,
                 mode='random', apply_pywt=False, site_meta_path='dataset/icecore/data_sources.csv',
                 use_spectral_features=1, cluster_result_path='cluster_results_cosine_zscore.csv',
                 residual_prediction=False, pred_seperate=None,
                 support_ratio=None, filter_cid = None):

        self.cluster_result_path = cluster_result_path
        self.residual_prediction = residual_prediction
        self.support_ratio = support_ratio
        self.pred_seperate = pred_seperate
        self.pred_seperate_list = None
        self.filter_cid=filter_cid
        if size == None:
            self.seq_len = 64
            self.label_len = 32
            self.pred_len = 16
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.x_data_path = x_data_path    # 【新增】
        self.y_data_path = y_data_path    # 【新增】
        self.site_meta_path = site_meta_path
        self.task_name = task_name
        self.is_pretraining = is_pretraining
        
        assert mode in ['random', 'region', 'cluster'], "mode 必须为 'random'、'region' 或 'cluster'"
        self.mode = mode
        self.apply_pywt = apply_pywt
        
        self.__read_data__()

    def __read_data__(self):
        # 【关键】双 Scaler 隔离
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.static_scaler = StandardScaler()

        # 1. 读取双变量文件
        df_x = pd.read_csv(os.path.join(self.root_path, self.x_data_path))
        df_y = pd.read_csv(os.path.join(self.root_path, self.y_data_path))
        
        # 2. 清理列名 & 取交集站点
        def clean_name(name):
            return str(name).strip().strip('"').strip("'").replace('  ', ' ')
        
        df_x = df_x.rename(columns={c: clean_name(c) for c in df_x.columns})
        df_y = df_y.rename(columns={c: clean_name(c) for c in df_y.columns})
        
        # 3. 加载聚类映射 (仅保留 common_sites)
        cluster_df_filter = pd.read_csv(self.cluster_result_path)
        cluster_df_filter.iloc[:, 0] = (cluster_df_filter.iloc[:, 0]
                                        .astype(str).str.strip().str.replace('"', '', regex=False))
        cluster_map_filter = cluster_df_filter.set_index(cluster_df_filter.columns[0])['Level_0'].to_dict()
        
        # 取交集 + 按年份对齐
        common_sites = sorted(set(df_x.columns) & set(df_y.columns) - {'year'})
                
        if self.filter_cid is not None:
            # 筛选有效站点，只保留 cid 对应簇
            filtered_sites = []
            for site in common_sites:
                clean_s = site.strip().replace('"','').replace("'",'')
                cid = cluster_map_filter.get(clean_s, -1)
                if cid == self.filter_cid:
                    filtered_sites.append(site)
            # common_sites = filtered_sites
            # print(f"筛选后只保留簇 {self.filter_cid} 的站点，共 {len(common_sites)} 个")

        df_x = df_x[['year'] + common_sites].sort_values('year').reset_index(drop=True)
        df_y = df_y[['year'] + common_sites].sort_values('year').reset_index(drop=True)
                
        # 4. 加载站点元数据
        df_meta = pd.read_csv(self.site_meta_path)
        df_meta['site_name_clean'] = df_meta['site'].astype(str).str.strip().str.replace('"', '', regex=False)
        site_meta_dict = df_meta.set_index('site_name_clean')[['Latitude', 'Longitude', 'Elevation']].to_dict('index')

        # ============ 辅助函数：单序列小波去噪 ============
        def _swt_denoise_single_series(series, wavelet='db2', level=1):
            valid_data = series.dropna().values
            N = len(valid_data)
            if N == 0:
                return valid_data
            
            pad_width = 1 if N % 2 != 0 else 0
            if pad_width > 0:
                x = np.pad(valid_data, (0, pad_width), mode='symmetric')
            else:
                x = valid_data.copy()
            
            if pywt.swt_max_level(len(x)) < level:
                return valid_data
            
            coeffs = pywt.swt(x, wavelet, level=level, start_level=0)
            cA1, cD1 = coeffs[0]
            
            sigma = median_abs_deviation(cD1, scale='normal')
            if sigma > 0:
                threshold = sigma * np.sqrt(2 * np.log(N))
                cD1_clean = pywt.threshold(cD1, value=threshold, mode='soft')
            else:
                cD1_clean = cD1
            
            clean_data = pywt.iswt([(cA1, cD1_clean)], wavelet)
            if pad_width > 0:
                clean_data = clean_data[:-pad_width]
            
            return clean_data


        def apply_pywt_to_dataframe(df, sites, wavelet='db2', level=1, verbose=True):
            if verbose:
                print(f"="*60 + f"\nApply pywt denoising (wavelet={wavelet}, level={level})...\n" + "="*60)
            
            df_copy = df.copy()
            for site in sites:
                series = df_copy[site]
                valid_idx = series.dropna().index
                clean_vals = _swt_denoise_single_series(series, wavelet, level)
                if len(valid_idx) == len(clean_vals):
                    df_copy.loc[valid_idx, site] = clean_vals
    
            if verbose:
                print(f"pywt done! Processed {len(sites)} sites.\n" + "="*60)
            
            return df_copy

        if self.apply_pywt:
            df_x_copy = apply_pywt_to_dataframe(
                df_x, common_sites, 
                wavelet='db2', level=1, verbose=True
            )
            # 对目标变量（如chem）去噪
            df_y_copy = apply_pywt_to_dataframe(
                df_y, common_sites, 
                wavelet='db2', level=1, verbose=False  # 只打印一次，避免重复
            )
        else:
            df_x_copy = df_x.copy()  # 不去噪时也要copy，保持接口一致
            df_y_copy = df_y.copy() 

        # 6. 筛选有效站点 (长度足够)
        valid_sites = []
        min_len = self.seq_len
        for site in common_sites:
            if len(df_x[[site]].dropna()) >= min_len and len(df_y[[site]].dropna()) >= min_len:
                valid_sites.append(site)

        # 7. 按 mode 划分 train/val/test (逻辑完全复用原版)
        train_sites, val_sites, test_sites = [], [], []
        np.random.seed(2030)
        
        if self.mode == 'cluster':
            print(f"Dividing per cluster (common_sites={len(valid_sites)})...")
            cluster_df = pd.read_csv(self.cluster_result_path)
            cluster_df.iloc[:, 0] = (cluster_df.iloc[:, 0]
                          .astype(str).str.strip().str.replace('"', '', regex=False))
            cluster_map = cluster_df.set_index(cluster_df.columns[0])['Level_0'].to_dict()
            
            cluster_dict = {}
            for s in valid_sites:
                clean_s = clean_name(s)
                cid = cluster_map.get(clean_s, -1)
                cluster_dict.setdefault(cid, []).append(s)
            
            for cid, c_sites in cluster_dict.items():
                np.random.shuffle(c_sites)
                n = len(c_sites)
                if n == 1:
                    train_sites.extend(c_sites)
                elif n == 2:
                    train_sites.append(c_sites[0]); test_sites.append(c_sites[1])
                else:
                    r_train = int(n * 0.6); r_rem = n - r_train
                    r_val = r_rem // 2; r_test = r_rem - r_val
                    train_sites.extend(c_sites[:r_train])
                    val_sites.extend(c_sites[r_train:r_train+r_val])
                    test_sites.extend(c_sites[r_train+r_val:])

            if self.filter_cid is not None:
                def keep_current_cid(sites):
                    kept = []
                    for s in sites:
                        cid = cluster_map_filter.get(clean_name(s), -1)
                        if cid == self.filter_cid:
                            kept.append(s)
                    return kept

                full_train_sites = train_sites.copy()
                full_val_sites = val_sites.copy()
                full_test_sites = test_sites.copy()

                train_sites = keep_current_cid(train_sites)
                val_sites = keep_current_cid(val_sites)
                test_sites = keep_current_cid(test_sites)
        
        # 打印划分日志 (仅 train 时)
        if self.set_type == 0:
            print("="*60)
            print(f"数据划分模式: {self.mode}")
            print("-" * 60)
            print(f"Train 站点分配 ({len(train_sites)}个): {train_sites}")
            print(f"Val 站点分配 ({len(val_sites)}个): {val_sites}")
            print(f"Test 站点分配 ({len(test_sites)}个): {test_sites}")

            # ===== 新增：各簇在 train/val/test 中的分布 =====
            cluster_df_log = pd.read_csv(self.cluster_result_path)
            cluster_df_log.iloc[:, 0] = (cluster_df_log.iloc[:, 0]
                                        .astype(str).str.strip()
                                        .str.replace('"', '', regex=False))
            cluster_map_log = cluster_df_log.set_index(
                cluster_df_log.columns[0])['Level_0'].to_dict()

            def count_by_cluster(sites):
                counter = {}
                for s in sites:
                    cid = cluster_map_log.get(s.strip().replace('"', ''), -1)
                    counter[cid] = counter.get(cid, 0) + 1
                return counter

            train_cnt = count_by_cluster(train_sites)
            val_cnt   = count_by_cluster(val_sites)
            test_cnt  = count_by_cluster(test_sites)
            all_cids  = sorted(set(train_cnt) | set(val_cnt) | set(test_cnt))

            print(f"{'簇ID':<8} {'Train':>6} {'Val':>6} {'Test':>6} {'合计':>6}")
            print("-" * 36)
            for cid in all_cids:
                t = train_cnt.get(cid, 0)
                v = val_cnt.get(cid, 0)
                s = test_cnt.get(cid, 0)
                label = f"Cluster {cid}" if cid != -1 else "未匹配"
                print(f"{label:<8} {t:>6} {v:>6} {s:>6} {t+v+s:>6}")
            print("-" * 36)
            print(f"{'合计':<8} {len(train_sites):>6} {len(val_sites):>6} "
            f"{len(test_sites):>6} {len(train_sites)+len(val_sites)+len(test_sites):>6}")
            # ===== 新增结束 =====
            print("="*60)

        # 8. 确定当前 flag 对应的站点
        if self.set_type == 0: self.current_sites = train_sites
        elif self.set_type == 1: self.current_sites = val_sites
        else: self.current_sites = test_sites

        
        # 9. 拟合 Scaler (仅用 train_sites)
        if self.scale:
            x_train = np.vstack([df_x_copy[[s]].dropna().values for s in train_sites])
            y_train = np.vstack([df_y_copy[[s]].dropna().values for s in train_sites])
            self.x_scaler.fit(x_train)
            self.y_scaler.fit(y_train)

        # 10. 预计算: 站点->簇映射 + 历史相关系数
        self.site_cluster_map = {}
        self.site_pearson_map = {}  # 【新增】历史 pearson(x,y)



        for site_idx,site in enumerate(self.current_sites):
            clean_s = clean_name(site)
            self.site_cluster_map[site_idx] = cluster_map_filter.get(clean_s, -1)
            # 计算全时段历史相关 (去趋势后)
            x_full = df_x_copy[site].dropna().values
            y_full = df_y_copy[site].dropna().values
            min_len = min(len(x_full), len(y_full))
            if min_len >= 30:  # 足够计算相关
                x_dt = detrend(x_full[:min_len], type='linear')
                y_dt = detrend(y_full[:min_len], type='linear')
                rho, _ = spearmanr(x_dt, y_dt)
                self.site_pearson_map[site] = rho if not np.isnan(rho) else 0.0

        # 11. 构建数据列表 (核心改动: x/y 分离但时间对齐)
        self.static_feat_list = []
        self.data_x_list = []   # 源变量 (输入)
        self.data_y_list = []   # 目标变量 (标签)
        self.data_stamp_list = []
        self.year_list = []
        self.valid_indices = []

        site_idx=0
        
        # 先 fit static_scaler (用 train_sites)
        train_static = []
        for s in train_sites:
            clean_s = clean_name(s)
            meta = site_meta_dict.get(clean_s, {'Latitude':0,'Longitude':0,'Elevation':0})
            lon_rad = np.radians(meta['Longitude'])
            train_static.append([meta['Latitude'], np.sin(lon_rad), np.cos(lon_rad), meta['Elevation']])
        if train_static:
            self.static_scaler.fit(np.array(train_static))
        
        for site in self.current_sites:
            clean_s = clean_name(site)
            meta = site_meta_dict.get(clean_s, {'Latitude':0,'Longitude':0,'Elevation':0})
            lon_rad = np.radians(meta['Longitude'])
            static_raw = np.array([[meta['Latitude'], np.sin(lon_rad), np.cos(lon_rad), meta['Elevation']]])
            static_scaled = self.static_scaler.transform(static_raw)[0]
            
            # 【关键】双变量对齐提取
            df_site_x = df_x_copy[['year', site]].dropna()
            df_site_y = df_y_copy[['year', site]].dropna()
            common_years = df_site_x.set_index('year').index.intersection(df_site_y.set_index('year').index)
            df_site_x = df_site_x.set_index('year').loc[common_years].reset_index()
            df_site_y = df_site_y.set_index('year').loc[common_years].reset_index()
            
            x_vals = df_site_x[[site]].values
            y_vals = df_site_y[[site]].values
            
            if self.scale:
                x_vals = self.x_scaler.transform(x_vals)
                y_vals = self.y_scaler.transform(y_vals)
            
            # 时间戳
            df_stamp = df_site_x[['year']].copy()
            df_stamp['year_norm'] = (df_stamp['year'] - df_stamp['year'].mean()) / df_stamp['year'].std()
            data_stamp = df_stamp[['year_norm']].values
            
            # 追加历史相关系数到 static_feat (末尾+1维)
            hist_corr = self.site_pearson_map[site]
            static_feat = np.concatenate([static_scaled, [hist_corr]], axis=0)  # [4+1=5]
            
            self.static_feat_list.append(static_feat)
            self.data_x_list.append(x_vals)   # 源变量
            self.data_y_list.append(y_vals)   # 目标变量
            self.data_stamp_list.append(data_stamp)
            self.year_list.append(df_site_x['year'].values)
            seq_count = len(x_vals) - self.seq_len + 1
            for i in range(max(0, seq_count)):
                self.valid_indices.append((site_idx, i))
            site_idx+=1


        if self.set_type == 0 and self.percent < 100:
            keep = int(len(self.valid_indices) * self.percent / 100)
            self.valid_indices = self.valid_indices[:keep]

        # 13. MAML support_len + 原型 (复用原版逻辑，仅用 x 变量构建原型)
        self.support_len_list = []
        for site_idx in range(len(self.data_x_list)):
            total = len(self.data_x_list[site_idx])
            if self.support_ratio is not None:
                sl = int(total * self.support_ratio)
                sl = max(sl, self.seq_len)
                sl = min(sl, total - self.seq_len)
            else:
                sl = 100
            self.support_len_list.append(sl)
        self.support_len = min(self.support_len_list) if self.support_len_list else 100
        
        # 原型仅用 x 变量构建 (源变量动态特征)
        self.prototypes = {0:[], 1:[], 2:[]}
        for site in train_sites:
            clean_s = clean_name(site)
            cid = cluster_map_filter.get(clean_s, -1)
            if cid in self.prototypes:
                df_proto = df_x_copy[['year', site]].dropna().set_index('year').loc[common_years].reset_index()
                vals = df_proto[[site]].values
                if self.scale: vals = self.x_scaler.transform(vals)
                if len(vals) >= self.support_len:
                    self.prototypes[cid].append(vals[:self.support_len].flatten())
        for cid in self.prototypes:
            if self.prototypes[cid]:
                self.prototypes[cid] = np.mean(self.prototypes[cid], axis=0)
            else:
                self.prototypes[cid] = np.zeros(self.support_len)

        # 残差预测用簇均值 (用 y 变量)
        if self.residual_prediction:
            self.cluster_time_means_x = {}
            self.cluster_time_means_y = {}
            for cid in range(3):
                c_sites = [s for s in train_sites if cluster_map_filter.get(clean_name(s), -1) == cid]
                if not c_sites: continue
                stack_x = np.stack([df_x_copy.set_index('year').loc[common_years, s].values for s in c_sites], axis=0)
                mean_data_x = stack_x.mean(axis=0)
                stack_y = np.stack([df_y_copy.set_index('year').loc[common_years, s].values for s in c_sites], axis=0)
                mean_data_y = stack_y.mean(axis=0)
                if self.scale:
                    mean_data_x = self.x_scaler.transform(mean_data_x.reshape(-1,1)).flatten()
                    mean_data_y = self.y_scaler.transform(mean_data_y.reshape(-1,1)).flatten()
                self.cluster_time_means_x[cid] = mean_data_x
                self.cluster_time_means_y[cid] = mean_data_y

    def __getitem__(self, index):
        site_idx = index
        site_name = self.current_sites[site_idx]
        site_support_len = self.support_len_list[site_idx]
        
        # 获取双变量数据 (已对齐)
        data_x = self.data_x_list[site_idx]  # 源变量 (输入)
        data_y = self.data_y_list[site_idx]  # 目标变量 (标签)
        data_stamp = self.data_stamp_list[site_idx]
        total_len = len(data_x)
        window = self.seq_len

        # 残差预测处理
        cluster_id = self.site_cluster_map.get(site_idx, 0)
        if self.residual_prediction:
            c_mean_x = self.cluster_time_means_x.get(cluster_id, np.zeros(total_len)).reshape(-1,1)
            c_mean_y = self.cluster_time_means_y.get(cluster_id, np.zeros(total_len)).reshape(-1,1)
            data_x_model = data_x - c_mean_x
            data_y_model = data_y - c_mean_y
        else:
            c_mean_x = np.zeros(total_len).reshape(-1, 1)
            c_mean_y = np.zeros(total_len).reshape(-1, 1)
            data_x_model, data_y_model = data_x, data_y

        # 【关键】同时间切片: support_x/query_x 来自 data_x, support_y/query_y 来自 data_y
        support_x, support_y, support_x_mark, support_y_mark = [], [], [], []
        for i in range(0, site_support_len - window + 1, 1):
            s_b, s_e = i, i + self.seq_len
            support_x.append(data_x_model[s_b:s_e])
            support_y.append(data_y_model[s_b:s_e])   # 【关键】同时间窗口的目标变量
            support_x_mark.append(data_stamp[s_b:s_e])
            support_y_mark.append(data_stamp[s_b:s_e])
        
        query_x, query_y, query_x_mark, query_y_mark, query_years = [], [], [], [], []
        query_x_mean = []
        query_y_mean = []
        q_start = site_support_len
        for i in range(q_start, total_len - window + 1, 1):
            s_b, s_e = i, i + self.seq_len
            query_x.append(data_x_model[s_b:s_e])
            query_y.append(data_y_model[s_b:s_e])     # 【关键】同时间窗口的目标变量
            query_x_mark.append(data_stamp[s_b:s_e])
            query_y_mark.append(data_stamp[s_b:s_e])
            query_years.append(self.year_list[site_idx][s_b])
            query_x_mean.append(c_mean_x[s_b : s_e])
            query_y_mean.append(c_mean_y[s_b : s_e])
            
        cluster_id = self.site_cluster_map.get(site_idx, 0)
        if self.filter_cid is None:
            one_hot = [1.0 if cluster_id == c else 0.0 for c in range(3)]
        
        # 构建 static_feat (已预计算 + 历史相关系数)
        static_feat = self.static_feat_list[site_idx]  # [geo(4) + hist_corr(1)]
        
        # 局部统计量 (用源变量 support 段计算)
        support_raw = data_x[:site_support_len].flatten()
        local_mean, local_std = np.mean(support_raw), np.std(support_raw) + 1e-6
        fft_p = np.abs(np.fft.rfft(support_raw - local_mean))**2
        split = max(1, len(fft_p)//5)
        local_f0 = fft_p[:split].sum() / (fft_p[split:].sum() + 1e-8)
        
        # 原型相似度 (用源变量)
        proto_raw = data_x[:self.support_len].flatten()
        sim_feats = []
        if self.filter_cid is None:
            for cid in range(3):
                proto = self.prototypes.get(cid, np.zeros(self.support_len))
                if len(proto_raw) < self.support_len or np.std(proto_raw)<1e-6 or np.std(proto)<1e-6:
                    sim_feats.append(0.0)
                else:
                    sim = np.corrcoef(proto_raw, proto)[0,1]
                    sim_feats.append(sim if not np.isnan(sim) else 0.0)
            maml_static = np.array(one_hot+sim_feats + static_feat.tolist() + [local_mean, local_std, local_f0], dtype=np.float32)
        else:
            maml_static = np.array(static_feat.tolist() + [local_mean, local_std, local_f0], dtype=np.float32)

        return (
            torch.tensor(np.array(support_x), dtype=torch.float32),
            torch.tensor(np.array(support_y), dtype=torch.float32),
            torch.tensor(np.array(support_x_mark), dtype=torch.float32),
            torch.tensor(np.array(support_y_mark), dtype=torch.float32),
            torch.tensor(np.array(query_x), dtype=torch.float32),
            torch.tensor(np.array(query_y), dtype=torch.float32),   # 【关键】目标变量标签
            torch.tensor(np.array(query_x_mark), dtype=torch.float32),
            torch.tensor(np.array(query_y_mark), dtype=torch.float32),
            torch.tensor(maml_static, dtype=torch.float32),
            torch.tensor(np.array(query_years), dtype=torch.float32),
            torch.tensor(np.array(query_x_mean), dtype=torch.float32),
            torch.tensor(np.array(query_y_mean), dtype=torch.float32)  
        )

    def __len__(self):
        return len(self.current_sites)

    def inverse_transform(self, data, var='y'):
        """反归一化: var='x' 用 x_scaler, var='y' 用 y_scaler"""
        if not self.scale:
            return data
        scaler = self.y_scaler if var == 'y' else self.x_scaler
        return scaler.inverse_transform(data)



class Dataset_IceCore_MS(Dataset):
    """
    跨变量预测专用 DataLoader:
    - 输入: x_csv (如 accum), y_csv (如 chem)
    - 输出: support_x/query_x 来自源变量，support_y/query_y 来自目标变量
    - 时间严格对齐: 同几年预测同几年 (非滞后外推)
    - static_feat 末尾追加: 该站点历史 pearson(x, y)
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='S', 
                 data_path=None,
                 x_data_path='accum.csv',      # 【新增】源变量文件
                 y_data_path='merged_chem_clean.csv',  # 【新增】目标变量文件
                 target='na', scale=True, timeenc=0, freq='y',
                 percent=100, task_name='long_term_forecast', is_pretraining=0,
                 mode='random', apply_pywt=False, site_meta_path='dataset/icecore/data_sources.csv',
                 use_spectral_features=1, cluster_result_path='cluster_results_cosine_zscore.csv',
                 residual_prediction=False, pred_seperate=None,
                 support_ratio=None, filter_cid = None):

        self.cluster_result_path = cluster_result_path
        self.residual_prediction = residual_prediction
        self.support_ratio = support_ratio
        self.pred_seperate = pred_seperate
        self.pred_seperate_list = None
        self.filter_cid=filter_cid
        if size == None:
            self.seq_len = 64
            self.label_len = 32
            self.pred_len = 16
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.x_data_path = x_data_path    # 【新增】
        self.y_data_path = y_data_path    # 【新增】
        self.site_meta_path = site_meta_path
        self.task_name = task_name
        self.is_pretraining = is_pretraining
        
        assert mode in ['random', 'region', 'cluster'], "mode 必须为 'random'、'region' 或 'cluster'"
        self.mode = mode
        self.apply_pywt = apply_pywt
        
        self.__read_data__()

    def __read_data__(self):
        # 【关键】双 Scaler 隔离
        self.accum_scaler = StandardScaler()
        self.chem_scaler = StandardScaler()
        self.static_scaler = StandardScaler()

        df_accum=pd.read_csv(os.path.join(self.root_path, 'accum_100y.csv')).set_index('year')
        df_chem = pd.read_csv(os.path.join(self.root_path, 'chem_clean.csv')).set_index('year')
        
        def clean_name(name):
            return str(name).strip().strip('"').strip("'").replace('  ', ' ')
        
        df_accum = df_accum.rename(columns={c: clean_name(c) for c in df_accum.columns})
        df_chem = df_chem.rename(columns={c: clean_name(c) for c in df_chem.columns})

        cluster_df_filter = pd.read_csv(self.cluster_result_path)
        cluster_df_filter.iloc[:, 0] = (cluster_df_filter.iloc[:, 0]
                                        .astype(str).str.strip().str.replace('"', '', regex=False))
        cluster_map_filter = cluster_df_filter.set_index(cluster_df_filter.columns[0])['Level_0'].to_dict()
        
        # 取交集 + 按年份对齐
        common_sites = sorted(set(df_accum.columns) & set(df_chem.columns) - {'year'})

        if self.filter_cid is not None:
            # 筛选有效站点，只保留 cid 对应簇
            filtered_sites = []
            for site in common_sites:
                clean_s = site.strip().replace('"','').replace("'",'')
                cid = cluster_map_filter.get(clean_s, -1)
                if cid == self.filter_cid:
                    filtered_sites.append(site)
            common_sites = filtered_sites
            # print(f"筛选后只保留簇 {self.filter_cid} 的站点，共 {len(common_sites)} 个")

        df_accum = df_accum.reset_index()
        df_chem = df_chem.reset_index()
        df_accum = df_accum[['year'] + common_sites].sort_values('year').reset_index(drop=True)
        df_chem = df_chem[['year'] + common_sites].sort_values('year').reset_index(drop=True)

        def _swt_denoise_single_series(series, wavelet='db2', level=1):
            valid_data = series.dropna().values
            N = len(valid_data)
            if N == 0:
                return valid_data
            
            pad_width = 1 if N % 2 != 0 else 0
            if pad_width > 0:
                x = np.pad(valid_data, (0, pad_width), mode='symmetric')
            else:
                x = valid_data.copy()
            
            if pywt.swt_max_level(len(x)) < level:
                return valid_data
            
            coeffs = pywt.swt(x, wavelet, level=level, start_level=0)
            cA1, cD1 = coeffs[0]
            
            sigma = median_abs_deviation(cD1, scale='normal')
            if sigma > 0:
                threshold = sigma * np.sqrt(2 * np.log(N))
                cD1_clean = pywt.threshold(cD1, value=threshold, mode='soft')
            else:
                cD1_clean = cD1
            
            clean_data = pywt.iswt([(cA1, cD1_clean)], wavelet)
            if pad_width > 0:
                clean_data = clean_data[:-pad_width]
            
            return clean_data


        def apply_pywt_to_dataframe(df, sites, wavelet='db2', level=1, verbose=True):
            if verbose:
                print(f"="*60 + f"\nApply pywt denoising (wavelet={wavelet}, level={level})...\n" + "="*60)
            
            df_copy = df.copy()
            for site in sites:
                series = df_copy[site]
                valid_idx = series.dropna().index
                clean_vals = _swt_denoise_single_series(series, wavelet, level)
                if len(valid_idx) == len(clean_vals):
                    df_copy.loc[valid_idx, site] = clean_vals
    
            if verbose:
                print(f"pywt done! Processed {len(sites)} sites.\n" + "="*60)
            
            return df_copy

        if self.apply_pywt:
            df_accum_pywt = apply_pywt_to_dataframe(
                df_accum, common_sites, 
                wavelet='db2', level=1, verbose=True
            )
            # 对目标变量（如chem）去噪
            df_chem_pywt = apply_pywt_to_dataframe(
                df_chem, common_sites, 
                wavelet='db2', level=1, verbose=False  # 只打印一次，避免重复
            )
        else:
            df_accum_pywt = df_accum.copy()  # 不去噪时也要copy，保持接口一致
            df_chem_pywt = df_chem.copy() 
                

                
        # 4. 加载站点元数据
        df_meta = pd.read_csv(self.site_meta_path)
        df_meta['site_name_clean'] = df_meta['site'].astype(str).str.strip().str.replace('"', '', regex=False)
        site_meta_dict = df_meta.set_index('site_name_clean')[['Latitude', 'Longitude', 'Elevation']].to_dict('index')

        
        # 6. 筛选有效站点 (长度足够)
        valid_sites = []
        min_len = self.seq_len+self.pred_len
        for site in common_sites:
            if len(df_accum[[site]].dropna()) >= min_len and len(df_chem[[site]].dropna()) >= min_len:
                valid_sites.append(site)

        # 7. 按 mode 划分 train/val/test (逻辑完全复用原版)
        train_sites, val_sites, test_sites = [], [], []
        np.random.seed(2030)
        
        if self.mode == 'cluster':
            print(f"Dividing per cluster (common_sites={len(valid_sites)})...")
            cluster_df = pd.read_csv(self.cluster_result_path)
            cluster_df.iloc[:, 0] = (cluster_df.iloc[:, 0]
                          .astype(str).str.strip().str.replace('"', '', regex=False))
            cluster_map = cluster_df.set_index(cluster_df.columns[0])['Level_0'].to_dict()
            
            cluster_dict = {}
            for s in valid_sites:
                clean_s = clean_name(s)
                cid = cluster_map.get(clean_s, -1)
                cluster_dict.setdefault(cid, []).append(s)
            
            for cid, c_sites in cluster_dict.items():
                np.random.shuffle(c_sites)
                n = len(c_sites)
                if n == 1:
                    train_sites.extend(c_sites)
                elif n == 2:
                    train_sites.append(c_sites[0]); test_sites.append(c_sites[1])
                else:
                    r_train = int(n * 0.6); r_rem = n - r_train
                    r_val = r_rem // 2; r_test = r_rem - r_val
                    train_sites.extend(c_sites[:r_train])
                    val_sites.extend(c_sites[r_train:r_train+r_val])
                    test_sites.extend(c_sites[r_train+r_val:])

            if self.filter_cid is not None:
                def keep_current_cid(sites):
                    kept = []
                    for s in sites:
                        cid = cluster_map_filter.get(clean_name(s), -1)
                        if cid == self.filter_cid:
                            kept.append(s)
                    return kept

                full_train_sites = train_sites.copy()
                full_val_sites = val_sites.copy()
                full_test_sites = test_sites.copy()

                train_sites = keep_current_cid(train_sites)
                val_sites = keep_current_cid(val_sites)
                test_sites = keep_current_cid(test_sites)
        
        # 打印划分日志 (仅 train 时)
        if self.set_type == 0:
            print("="*60)
            print(f"数据划分模式: {self.mode}")
            print("-" * 60)
            print(f"Train 站点分配 ({len(train_sites)}个): {train_sites}")
            print(f"Val 站点分配 ({len(val_sites)}个): {val_sites}")
            print(f"Test 站点分配 ({len(test_sites)}个): {test_sites}")

            # ===== 新增：各簇在 train/val/test 中的分布 =====
            cluster_df_log = pd.read_csv(self.cluster_result_path)
            cluster_df_log.iloc[:, 0] = (cluster_df_log.iloc[:, 0]
                                        .astype(str).str.strip()
                                        .str.replace('"', '', regex=False))
            cluster_map_log = cluster_df_log.set_index(
                cluster_df_log.columns[0])['Level_0'].to_dict()

            def count_by_cluster(sites):
                counter = {}
                for s in sites:
                    cid = cluster_map_log.get(s.strip().replace('"', ''), -1)
                    counter[cid] = counter.get(cid, 0) + 1
                return counter

            train_cnt = count_by_cluster(train_sites)
            val_cnt   = count_by_cluster(val_sites)
            test_cnt  = count_by_cluster(test_sites)
            all_cids  = sorted(set(train_cnt) | set(val_cnt) | set(test_cnt))

            print(f"{'簇ID':<8} {'Train':>6} {'Val':>6} {'Test':>6} {'合计':>6}")
            print("-" * 36)
            for cid in all_cids:
                t = train_cnt.get(cid, 0)
                v = val_cnt.get(cid, 0)
                s = test_cnt.get(cid, 0)
                label = f"Cluster {cid}" if cid != -1 else "未匹配"
                print(f"{label:<8} {t:>6} {v:>6} {s:>6} {t+v+s:>6}")
            print("-" * 36)
            print(f"{'合计':<8} {len(train_sites):>6} {len(val_sites):>6} "
            f"{len(test_sites):>6} {len(train_sites)+len(val_sites)+len(test_sites):>6}")
            # ===== 新增结束 =====
            print("="*60)

        # 8. 确定当前 flag 对应的站点
        if self.set_type == 0: self.current_sites = train_sites
        elif self.set_type == 1: self.current_sites = val_sites
        else: self.current_sites = test_sites


        
        # 9. 拟合 Scaler (仅用 train_sites)
        if self.scale:
            df_accum_scaled = df_accum_pywt.copy()
            df_chem_scaled = df_chem_pywt.copy()
            accum_train = df_accum_pywt[train_sites].dropna().values   # shape: N_train x num_train_sites
            self.accum_scaler.fit(accum_train)
            df_accum_scaled[train_sites] = self.accum_scaler.transform(df_accum_scaled[train_sites])
            non_train_sites = [c for c in common_sites if c not in train_sites]
            for col in non_train_sites:
                # 用训练集 scaler 的 mean_ / scale_ 的平均值做标准化
                col_mean = np.mean(self.accum_scaler.mean_)
                col_std  = np.mean(self.accum_scaler.scale_) + 1e-6
                df_accum_scaled[col] = (df_accum_scaled[col] - col_mean) / col_std

            chem_train = df_chem_pywt[train_sites].dropna().values   # shape: N_train x num_train_sites
            self.chem_scaler.fit(chem_train)
            df_chem_scaled[train_sites] = self.chem_scaler.transform(df_chem_scaled[train_sites])
            for col in non_train_sites:
                # 用训练集 scaler 的 mean_ / scale_ 的平均值做标准化
                col_mean = np.mean(self.chem_scaler.mean_)
                col_std  = np.mean(self.chem_scaler.scale_) + 1e-6
                df_chem_scaled[col] = (df_chem_scaled[col] - col_mean) / col_std

            df_accum_pywt = df_accum_scaled
            df_chem_pywt = df_chem_scaled
        else:
            df_accum_pywt=df_accum_pywt
            df_chem_pywt=df_chem_pywt

        # 10. 预计算: 站点->簇映射 + 历史相关系数
        self.site_cluster_map = {}
        self.site_pearson_map = {}  # 【新增】历史 pearson(x,y)
        # 1. 读取双变量文件
        df_copy = pd.concat([
            df_accum_pywt.set_index('year'),
            df_chem_pywt.set_index('year')
        ], axis=1)
        # if self.target == 'accum':
        #     df_y_copy = df_accum_pywt.set_index('year')
        # else:
        #     df_y_copy = df_chem_pywt.set_index('year')

        df_copy=df_copy.reset_index()
        # df_y_copy=df_y_copy.reset_index()

        for site_idx,site in enumerate(self.current_sites):
            clean_s = clean_name(site)
            self.site_cluster_map[site_idx] = cluster_map_filter.get(clean_s, -1)
            # 计算全时段历史相关 (去趋势后)
            accum_full = df_accum_pywt[site].dropna().values
            chem_full = df_chem_pywt[site].dropna().values
            min_len = min(len(accum_full), len(chem_full))
            if min_len >= 30:  # 足够计算相关
                accum_dt = detrend(accum_full[:min_len], type='linear')
                chem_dt = detrend(chem_full[:min_len], type='linear')
                rho, _ = spearmanr(accum_dt, chem_dt)
                self.site_pearson_map[site] = rho if not np.isnan(rho) else 0.0

        # 11. 构建数据列表 (核心改动: x/y 分离但时间对齐)
        self.static_feat_list = []
        self.data_x_list = []   # 源变量 (输入)
        self.data_y_list = []   # 目标变量 (标签)
        self.data_stamp_list = []
        self.year_list = []
        self.valid_indices = []

        site_idx=0
        
        # 先 fit static_scaler (用 train_sites)
        train_static = []
        for s in train_sites:
            clean_s = clean_name(s)
            meta = site_meta_dict.get(clean_s, {'Latitude':0,'Longitude':0,'Elevation':0})
            lon_rad = np.radians(meta['Longitude'])
            train_static.append([meta['Latitude'], np.sin(lon_rad), np.cos(lon_rad), meta['Elevation']])
        if train_static:
            self.static_scaler.fit(np.array(train_static))
        
        for site in self.current_sites:
            clean_s = clean_name(site)
            meta = site_meta_dict.get(clean_s, {'Latitude':0,'Longitude':0,'Elevation':0})
            lon_rad = np.radians(meta['Longitude'])
            static_raw = np.array([[meta['Latitude'], np.sin(lon_rad), np.cos(lon_rad), meta['Elevation']]])
            static_scaled = self.static_scaler.transform(static_raw)[0]

            data_vals = df_copy[[site]].values
            
            # 时间戳
            df_stamp = df_copy[['year']].copy()
            df_stamp['year_norm'] = (df_stamp['year'] - df_stamp['year'].mean()) / df_stamp['year'].std()
            data_stamp = df_stamp[['year_norm']].values
            
            # 追加历史相关系数到 static_feat (末尾+1维)
            hist_corr = self.site_pearson_map[site]
            static_feat = np.concatenate([static_scaled, [hist_corr]], axis=0)  # [4+1=5]
            
            self.static_feat_list.append(static_feat)
            self.data_x_list.append(data_vals)   # 源变量
            self.data_y_list.append(data_vals)   # 目标变量
            self.data_stamp_list.append(data_stamp)
            self.year_list.append(df_copy['year'].values)
            seq_count = len(data_vals) - self.seq_len - self.pred_len + 1
            for i in range(max(0, seq_count)):
                self.valid_indices.append((site_idx, i))
            site_idx+=1


        if self.set_type == 0 and self.percent < 100:
            keep = int(len(self.valid_indices) * self.percent / 100)
            self.valid_indices = self.valid_indices[:keep]

        # 13. MAML support_len + 原型 (复用原版逻辑，仅用 x 变量构建原型)
        self.support_len_list = []
        for site_idx in range(len(self.data_x_list)):
            total = len(self.data_x_list[site_idx])
            if self.support_ratio is not None:
                sl = int(total * self.support_ratio)
                sl = max(sl, self.seq_len + self.pred_len)
                sl = min(sl, total - self.seq_len- self.pred_len)
                sl = max(sl, self.seq_len + self.pred_len)
            else:
                sl = 100
            self.support_len_list.append(sl)
        self.support_len = min(self.support_len_list) if self.support_len_list else 100
        
        # 原型仅用 x 变量构建 (源变量动态特征)
        self.prototypes = {0:[], 1:[], 2:[]}
        if self.target=='accum':
            df_for_proto=df_accum_pywt
        else:
            df_for_proto=df_chem_pywt

        for site in train_sites:
            clean_s = clean_name(site)
            cid = cluster_map_filter.get(clean_s, -1)
            if cid in self.prototypes:
                df_proto = df_for_proto[['year', site]]
                vals = df_proto[[site]].values
                # if self.scale: vals = self.x_scaler.transform(vals)
                if len(vals) >= self.support_len:
                    self.prototypes[cid].append(vals[:self.support_len].flatten())
        for cid in self.prototypes:
            if self.prototypes[cid]:
                self.prototypes[cid] = np.mean(self.prototypes[cid], axis=0)
            else:
                self.prototypes[cid] = np.zeros(self.support_len)

        if self.residual_prediction:
            self.cluster_time_means = {}
            for cid in range(3):
                c_sites = [s for s in train_sites if cluster_map_filter.get(clean_name(s), -1) == cid]
                if not c_sites: continue
                stack_data = np.stack([df_copy.set_index('year').loc[:, s].values for s in c_sites], axis=0)
                mean_data = stack_data.mean(axis=0)
                self.cluster_time_means[cid] = mean_data

    def __getitem__(self, index):
        site_idx = index
        site_name = self.current_sites[site_idx]
        site_support_len = self.support_len_list[site_idx]
        
        # 获取双变量数据 (已对齐)
        data_x = self.data_x_list[site_idx]  # 源变量 (输入)
        data_y = self.data_y_list[site_idx]  # 目标变量 (标签)
        data_stamp = self.data_stamp_list[site_idx]
        total_len = len(data_x)
        window = self.seq_len+self.pred_len

        # 残差预测处理
        cluster_id = self.site_cluster_map.get(site_idx, 0)
        if self.residual_prediction:
            c_mean = self.cluster_time_means.get(cluster_id, np.zeros(total_len))
            data_x_model = data_x - c_mean
            data_y_model = data_y - c_mean
        else:
            data_x_model, data_y_model = data_x, data_y

        # 【关键】同时间切片: support_x/query_x 来自 data_x, support_y/query_y 来自 data_y
        support_x, support_y, support_x_mark, support_y_mark = [], [], [], []
        for i in range(0, site_support_len - window + 1, 1):
            s_begin = i
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            support_x.append(data_x_model[s_begin:s_end, :])  # 取 seq_len x 2
            support_y.append(data_y_model[r_begin:r_end, :])  # 取 seq_len 的 target# 【关键】同时间窗口的目标变量
            support_x_mark.append(data_stamp[s_begin:s_end])
            support_y_mark.append(data_stamp[r_begin + self.label_len : r_end])
        
        query_x, query_y, query_x_mark, query_y_mark, query_years = [], [], [], [], []
        query_x_mean = []
        query_y_mean = []
        q_start = site_support_len-self.seq_len
        for i in range(q_start, total_len - window + 1, 1):
            s_begin = i
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            query_x.append(data_x_model[s_begin:s_end, :])
            query_y.append(data_y_model[r_begin:r_end, :])    # 【关键】同时间窗口的目标变量
            query_x_mark.append(data_stamp[s_begin:s_end])
            query_y_mark.append(data_stamp[r_begin:r_end])
            query_years.append(self.year_list[site_idx][r_begin + self.label_len : r_end])
            query_x_mean.append(c_mean[s_begin : s_end,:])
            query_y_mean.append(c_mean[r_begin + self.label_len : r_end,:])
            
        cluster_id = self.site_cluster_map.get(site_idx, 0)
        if self.filter_cid is None:
            one_hot = [1.0 if cluster_id == c else 0.0 for c in range(3)]
        
        # 构建 static_feat (已预计算 + 历史相关系数)
        static_feat = self.static_feat_list[site_idx]  # [geo(4) + hist_corr(1)]
        
        if self.target=='accum':
            data_local=data_x[:site_support_len, 0]
        else:
            data_local=data_x[:site_support_len, 1]

        # 局部统计量 (用源变量 support 段计算)
        support_raw = data_local[:site_support_len].flatten()
        local_mean, local_std = np.mean(support_raw), np.std(support_raw) + 1e-6
        fft_p = np.abs(np.fft.rfft(support_raw - local_mean))**2
        split = max(1, len(fft_p)//5)
        local_f0 = fft_p[:split].sum() / (fft_p[split:].sum() + 1e-8)
        
        # 原型相似度 (用源变量)
        proto_raw = data_local[:self.support_len].flatten()
        sim_feats = []
        if self.filter_cid is None:
            for cid in range(3):
                proto = self.prototypes.get(cid, np.zeros(self.support_len))
                if len(proto_raw) < self.support_len or np.std(proto_raw)<1e-6 or np.std(proto)<1e-6:
                    sim_feats.append(0.0)
                else:
                    sim = np.corrcoef(proto_raw, proto)[0,1]
                    sim_feats.append(sim if not np.isnan(sim) else 0.0)
            maml_static = np.array(one_hot+sim_feats + static_feat.tolist() + [local_mean, local_std, local_f0], dtype=np.float32)
        else:
            maml_static = np.array(static_feat.tolist() + [local_mean, local_std, local_f0], dtype=np.float32)
        return (
            torch.tensor(np.array(support_x), dtype=torch.float32),
            torch.tensor(np.array(support_y), dtype=torch.float32),
            torch.tensor(np.array(support_x_mark), dtype=torch.float32),
            torch.tensor(np.array(support_y_mark), dtype=torch.float32),
            torch.tensor(np.array(query_x), dtype=torch.float32),
            torch.tensor(np.array(query_y), dtype=torch.float32),   # 【关键】目标变量标签
            torch.tensor(np.array(query_x_mark), dtype=torch.float32),
            torch.tensor(np.array(query_y_mark), dtype=torch.float32),
            torch.tensor(maml_static, dtype=torch.float32),
            torch.tensor(np.array(query_years), dtype=torch.float32),
            torch.tensor(np.array(query_x_mean), dtype=torch.float32),
            torch.tensor(np.array(query_y_mean), dtype=torch.float32)  
        )

    def __len__(self):
        return len(self.current_sites)

    def inverse_transform(self, data, var='y'):
        """反归一化: var='x' 用 x_scaler, var='y' 用 y_scaler"""
        if not self.scale:
            return data
        scaler = self.y_scaler if var == 'y' else self.x_scaler
        return scaler.inverse_transform(data)



