import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def normalize1(x):
    return (x - x.min()) / (x.max() - x.min())

def normalize2(x, y):
    return (y - x.min()) / (x.max() - x.min())


class EvalDataset1_nec(Dataset):
    def __init__(self, knnfile, testfile):
        super(EvalDataset1_nec, self).__init__()
        self.knnfile = knnfile
        self.testfile = testfile
        self.data = list(range(1, 57)) + [57]

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

        self.nec1 = self.df['nec'] == 1
        self.df1 = self.df[self.nec1]
        self.np_df1 = self.df1.values

    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[:, 54:][idx]) - 1.
        return testdata1, labeldata1

    def __len__(self):
        length = len(self.np_df1)
        return length


class EvalDataset2_nec(Dataset):
    def __init__(self, knnfile, testfile):
        super(EvalDataset2_nec, self).__init__()
        self.knnfile = knnfile
        self.testfile = testfile
        self.data = list(range(1, 57)) + [57]

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

        self.nec2 = self.df['nec'] == 2
        self.df2 = self.df[self.nec2]
        self.np_df2 = self.df2.values

    def __getitem__(self, idx):
        testdata2 = torch.FloatTensor(self.np_df2[:, :54][idx])
        labeldata2 = torch.FloatTensor(self.np_df2[:, 54:][idx]) - 1.
        return testdata2, labeldata2

    def __len__(self):
        length = len(self.np_df2)
        return length


class EvalDataset1_necip(Dataset):
    def __init__(self, knnfile, testfile):
        super(EvalDataset1_necip, self).__init__()
        self.knnfile = knnfile
        self.testfile = testfile
        self.data = list(range(1, 57)) + [58]

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

        self.necip1 = self.df['necip'] == 1
        self.df1 = self.df[self.necip1]
        self.np_df1 = self.df1.values

    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[:, 54:][idx]) - 1.
        return testdata1, labeldata1

    def __len__(self):
        length = len(self.np_df1)
        return length


class EvalDataset2_necip(Dataset):
    def __init__(self, knnfile, testfile):
        super(EvalDataset2_necip, self).__init__()
        self.knnfile = knnfile
        self.testfile = testfile
        self.data = list(range(1, 57)) + [58]

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

        self.necip2 = self.df['necip'] == 2
        self.df2 = self.df[self.necip2]
        self.np_df2 = self.df2.values

    def __getitem__(self, idx):
        testdata2 = torch.FloatTensor(self.np_df2[:, :54][idx])
        labeldata2 = torch.FloatTensor(self.np_df2[:, 54:][idx]) - 1.
        return testdata2, labeldata2

    def __len__(self):
        length = len(self.np_df2)
        return length


class EvalDataset1_sip(Dataset):
    def __init__(self, knnfile, testfile):
        super(EvalDataset1_sip, self).__init__()
        self.knnfile = knnfile
        self.testfile = testfile
        self.data = list(range(1, 57)) + [59]

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

        #self.df['acl']=0

        self.sip1 = self.df['sip'] == 1
        self.df1 = self.df[self.sip1]
        self.np_df1 = self.df1.values


    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[:, 54:][idx]) - 1.
        return testdata1, labeldata1

    def __len__(self):
        length = len(self.np_df1)
        return length


class EvalDataset2_sip(Dataset):
    def __init__(self, knnfile, testfile):
        super(EvalDataset2_sip, self).__init__()
        self.knnfile = knnfile
        self.testfile = testfile
        self.data = list(range(1, 57)) + [59]

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

        self.sip2 = self.df['sip'] == 2
        self.df2 = self.df[self.sip2]
        self.np_df2 = self.df2.values


    def __getitem__(self, idx):
        testdata2 = torch.FloatTensor(self.np_df2[:, :54][idx])
        labeldata2 = torch.FloatTensor(self.np_df2[:, 54:][idx]) - 1.
        return testdata2, labeldata2

    def __len__(self):
        length = len(self.np_df2)
        return length


class EvalDataset1_ip(Dataset):
    def __init__(self, knnfile, testfile):
        super(EvalDataset1_ip, self).__init__()
        self.knnfile = knnfile
        self.testfile = testfile
        self.data = list(range(1, 57)) + [60]

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

        self.ip1 = self.df['ip'] == 1
        self.df1 = self.df[self.ip1]
        self.np_df1 = self.df1.values

    def __getitem__(self, idx):
        testdata2 = torch.FloatTensor(self.np_df1[:, :54][idx])
        labeldata2 = torch.FloatTensor(self.np_df1[:, 54:][idx]) - 1.
        return testdata2, labeldata2

    def __len__(self):
        length = len(self.np_df1)
        return length


class EvalDataset2_ip(Dataset):
    def __init__(self, knnfile, testfile):
        super(EvalDataset2_ip, self).__init__()
        self.knnfile = knnfile
        self.testfile = testfile
        self.data = list(range(1, 57)) + [60]

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

        self.ip2 = self.df['ip'] == 2
        self.df2 = self.df[self.ip2]
        self.np_df2 = self.df2.values

    def __getitem__(self, idx):
        testdata2 = torch.FloatTensor(self.np_df2[:, :54][idx])
        labeldata2 = torch.FloatTensor(self.np_df2[:, 54:][idx]) - 1.
        return testdata2, labeldata2

    def __len__(self):
        length = len(self.np_df2)
        return length

