import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def normalize1(x):
    return (x - x.min()) / (x.max() - x.min())

def normalize2(x, y):
    return (y - x.min()) / (x.max() - x.min())


class EvalDataset_nec_both(Dataset):
    def __init__(self, knnfile, testfile):
        super(EvalDataset_nec_both, self).__init__()
        self.knnfile = knnfile
        self.testfile = testfile
        self.data = list(range(1, 59))

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)
        self.df_B = pd.read_csv(self.testfile, index_col = 0, usecols = self.data)
        # self.df_B = self.df_B.dropna()

        self.df = self.df_A.drop(['svamal'], axis=1)
        self.test = self.df_B.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = ['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
                self.test[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
                self.test[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)
                self.test[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train2 = self.test.loc[:, :'eythtran']
        self.df_test2 = self.test.loc[:, 'pdad':]

        self.df_train_norm = normalize2(self.df_train, self.df_train2)
        # self.df_train_norm = normalize1(self.df_train)

        self.df = self.df_train_norm.join(self.df_test2)
        # self.df = self.df.fillna(0)

        self.np_df1 = self.df.values

    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df1[:, 55:56][idx]) - 1.

        return testdata1, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df1)
        return length


class EvalDataset_necip_both(Dataset):
    def __init__(self, knnfile, testfile):
        super(EvalDataset_necip_both, self).__init__()
        self.knnfile = knnfile
        self.testfile = testfile
        self.data = list(range(1, 59))

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)
        self.df_B = pd.read_csv(self.testfile, index_col = 0, usecols = self.data)

        self.df = self.df_A.drop(['svamal'], axis=1)
        self.test = self.df_B.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = ['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
                self.test[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
                self.test[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)
                self.test[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train2 = self.test.loc[:, :'eythtran']
        self.df_test2 = self.test.loc[:, 'pdad':]

        self.df_train_norm = normalize2(self.df_train, self.df_train2)
        self.df = self.df_train_norm.join(self.df_test2)
        # self.df_train_norm = normalize1(self.df_train)
        # self.df = self.df_train_norm.join(self.df_test)

        self.np_df1 = self.df.values

    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df1[:, 55:56][idx]) - 1.

        return testdata1, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df1)
        return length


class EvalDataset_sip_both(Dataset):
    def __init__(self, knnfile, testfile):
        super(EvalDataset_sip_both, self).__init__()
        self.knnfile = knnfile
        self.testfile = testfile
        self.data = list(range(1, 58)) + [59]

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)
        self.df_B = pd.read_csv(self.testfile, index_col = 0, usecols = self.data)

        self.df = self.df_A.drop(['svamal'], axis=1)
        self.test = self.df_B.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = ['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
                self.test[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
                self.test[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)
                self.test[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train2 = self.test.loc[:, :'eythtran']
        self.df_test2 = self.test.loc[:, 'pdad':]

        self.df_train_norm = normalize2(self.df_train, self.df_train2)

        self.df = self.df_train_norm.join(self.df_test2)

        self.np_df1 = self.df.values

    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df1[:, 55:56][idx]) - 1.

        return testdata1, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df1)
        return length


class EvalDataset_nec_i(Dataset):
    def __init__(self, knnfile, testfile):
        super(EvalDataset_nec_i, self).__init__()
        self.knnfile = knnfile
        self.testfile = testfile
        self.data = list(range(1, 58)) + [59]

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)
        self.df_B = pd.read_csv(self.testfile, index_col = 0, usecols = self.data)

        self.df = self.df_A.drop(['svamal'], axis=1)
        self.test = self.df_B.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = ['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
                self.test[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
                self.test[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)
                self.test[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train2 = self.test.loc[:, :'eythtran']
        self.df_test2 = self.test.loc[:, 'pdad':]

        self.df_train_norm = normalize2(self.df_train, self.df_train2)

        self.df = self.df_train_norm.join(self.df_test2)

        self.np_df1 = self.df.values

    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df1[:, 55:56][idx]) - 1.

        return testdata1, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df1)
        return length


class EvalDataset_ip_both(Dataset):
    def __init__(self, knnfile, testfile):
        super(EvalDataset_ip_both, self).__init__()
        self.knnfile = knnfile
        self.testfile = testfile
        self.data = list(range(1, 58)) + [60]

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)
        self.df_B = pd.read_csv(self.testfile, index_col = 0, usecols = self.data)

        self.df = self.df_A.drop(['svamal'], axis=1)
        self.test = self.df_B.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = ['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
                self.test[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
                self.test[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)
                self.test[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train2 = self.test.loc[:, :'eythtran']
        self.df_test2 = self.test.loc[:, 'pdad':]

        self.df_train_norm = normalize2(self.df_train, self.df_train2)

        self.df = self.df_train_norm.join(self.df_test2)

        self.np_df1 = self.df.values

    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df1[:, 55:56][idx]) - 1.

        return testdata1, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df1)
        return length
