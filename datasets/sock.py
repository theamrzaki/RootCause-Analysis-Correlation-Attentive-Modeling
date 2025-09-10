import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class sock:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.num_vars = options['num_vars']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.shuffle = options['shuffle']

    def generate_example(self):
        # load data
        df_labels = pd.read_csv(os.path.join(self.data_dir, "labels.csv"))
        df_train = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        df_test = pd.read_csv(os.path.join(self.data_dir, "test.csv"))

        # same downsampling as msds
        df_train, df_test = df_train.values[::, 1:], df_test.values[::10, 1:]
        df_labels = df_labels.values[::10, 1:]   # drop "index" column

        """
        unique, counts = np.unique(labels, return_counts=True)
        total = counts.sum()
        percentages = {u: (c / total) * 100 for u, c in zip(unique, counts)}
        """

        # collapse multi-labels into binary anomaly indicator
        labels = np.max(df_labels, axis=1)

        # prepare normal train segments
        x_n_list = []
        for i in range(0, len(df_train), 10000):
            if i + 10000 < len(df_train):
                x_n_list.append(df_train[i:i + 10000])

        # prepare test anomaly segments
        test_x_lst, label_lst = [], []
        for i in np.where(labels == 1)[0]:
            if i - 2 * self.window_size > 0 and i + self.window_size < len(df_test):
                if sum(labels[i - 2 * self.window_size:i]) == 0:  # ensure clean context
                    test_x_lst.append(df_test[i-2*self.window_size:i + self.window_size])
                    label_lst.append(df_labels[i-2*self.window_size:i + self.window_size])

        # scale
        scaler = StandardScaler()#MinMaxScaler
        scaler.fit(np.concatenate(x_n_list, axis=0))
        x_n_list = [scaler.transform(i) for i in x_n_list]
        test_x_lst = [scaler.transform(i) for i in test_x_lst]

        # pack into data_dict
        self.data_dict['x_n_list'] = np.array(x_n_list)
        if self.shuffle:
            np.random.seed(self.seed)
            indices = np.random.permutation(len(self.data_dict['x_n_list']))
            self.data_dict['x_n_list'] = self.data_dict['x_n_list'][indices]
        self.data_dict['x_ab_list'] = np.array(test_x_lst)
        self.data_dict['label_list'] = np.array(label_lst)

    def save_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        np.save(os.path.join(self.data_dir, 'x_n_list'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'label_list'), self.data_dict['label_list'])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'), allow_pickle=False)
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'), allow_pickle=True)
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'), allow_pickle=True)
