import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def normalize1(x):
    return (x - x.min()) / (x.max() - x.min())


class TrainDataset_nec_both(Dataset):
    def __init__(self, knnfile):
        super(TrainDataset_nec_both, self).__init__()

        self.knnfile = knnfile

        self.data = list(range(1, 59))
        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)
        self.df = self.df_A.drop(['svamal'], axis=1)
        self.data_ = self.df.copy(deep = True)

        self.mode_feature = []#['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)
        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['nec'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['nec'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values

        self.df2_train = self.np_df2[:700, :]

        for i in range(10):
            self.df2_train = np.concatenate((self.df2_train, self.np_df2[:700, :]))

        print(self.df.info())

    def __getitem__(self, idx):

        t = np.concatenate((self.np_df1[1000:8000, :54], self.df2_train[:7000, :54]), axis=0)
        l1 = np.concatenate((self.np_df1[1000:8000, 54], self.df2_train[:7000, 54]), axis=0)
        l2 = np.concatenate((self.np_df1[1000:8000, 55], self.df2_train[:7000, 55]), axis=0)

        traindata = torch.as_tensor(t[idx]).float()
        labeldata1 = torch.as_tensor(l1[idx]).float() - 1.
        labeldata2 = torch.as_tensor(l2[idx]).float() - 1.

        return traindata, labeldata1, labeldata2

    def __len__(self):
        return 14000


class EvalDataset1_nec_both(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset1_nec_both, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 59))

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)

        self.df = self.df_A.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = []#['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)

        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['nec'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values

    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[8000:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[8000:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df1[8000:, 55:56][idx]) - 1.

        return testdata1, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df1[8000:, :])
        return length


class EvalDataset2_nec_both(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset2_nec_both, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 59))

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)

        self.df = self.df_A.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = []#['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)

        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['nec'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['nec'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values

        self.df2_test = self.np_df2[700:, :]

        for i in range(90):
            self.df2_test = np.concatenate((self.df2_test, self.np_df2[700:, :]))


    def __getitem__(self, idx):

        testdata2 = torch.FloatTensor(self.np_df2[700:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df2[700:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df2[700:, 55:56][idx]) - 1.

        return testdata2, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df2[700:, :])
        return length


class TrainDataset_necip_both(Dataset):
    def __init__(self, knnfile):
        super(TrainDataset_necip_both, self).__init__()

        self.knnfile = knnfile

        self.data = list(range(1, 59))
        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)
        self.df = self.df_A.drop(['svamal'], axis=1)
        self.data_ = self.df.copy(deep = True)

        self.mode_feature = []#['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)
        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['necip'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['necip'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values

        self.df2_train = self.np_df2[:380, :]

        for i in range(20):
            self.df2_train = np.concatenate((self.df2_train, self.np_df2[:380, :]))

        print(self.df.info())

    def __getitem__(self, idx):

        t = np.concatenate((self.np_df1[1000:8000, :54], self.df2_train[:7000, :54]), axis=0)
        l1 = np.concatenate((self.np_df1[1000:8000, 54], self.df2_train[:7000, 54]), axis=0)
        l2 = np.concatenate((self.np_df1[1000:8000, 55], self.df2_train[:7000, 55]), axis=0)

        traindata = torch.as_tensor(t[idx]).float()
        labeldata1 = torch.as_tensor(l1[idx]).float() - 1.
        labeldata2 = torch.as_tensor(l2[idx]).float() - 1.

        return traindata, labeldata1, labeldata2

    def __len__(self):
        return 14000


class EvalDataset1_necip_both(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset1_necip_both, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 59))

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)

        self.df = self.df_A.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = []#['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)

        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['necip'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values

    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[8000:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[8000:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df1[8000:, 55:56][idx]) - 1.

        return testdata1, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df1[8000:, :])
        return length


class EvalDataset2_necip_both(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset2_necip_both, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 59))

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)

        self.df = self.df_A.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = []#['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)

        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['necip'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['necip'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values

        self.df2_test = self.np_df2[380:, :]

        for i in range(90):
            self.df2_test = np.concatenate((self.df2_test, self.np_df2[380:, :]))


    def __getitem__(self, idx):

        testdata2 = torch.FloatTensor(self.np_df2[380:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df2[380:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df2[380:, 55:56][idx]) - 1.

        return testdata2, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df2[380:, :])
        return length


class TrainDataset_sip_both(Dataset):
    def __init__(self, knnfile):
        super(TrainDataset_sip_both, self).__init__()

        self.knnfile = knnfile

        self.data = list(range(1, 58)) + [59]
        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)
        self.df = self.df_A.drop(['svamal'], axis=1)
        self.data_ = self.df.copy(deep = True)

        self.mode_feature = []#['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)
        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['sip'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['sip'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values

        self.df2_train = self.np_df2[:135, :]

        for i in range(52):
            self.df2_train = np.concatenate((self.df2_train, self.np_df2[:135, :]))

        print(self.df.info())

    def __getitem__(self, idx):

        t = np.concatenate((self.np_df1[1000:8000, :54], self.df2_train[:7000, :54]), axis=0)
        l1 = np.concatenate((self.np_df1[1000:8000, 54], self.df2_train[:7000, 54]), axis=0)
        l2 = np.concatenate((self.np_df1[1000:8000, 55], self.df2_train[:7000, 55]), axis=0)

        traindata = torch.as_tensor(t[idx]).float()
        labeldata1 = torch.as_tensor(l1[idx]).float() - 1.
        labeldata2 = torch.as_tensor(l2[idx]).float() - 1.

        return traindata, labeldata1, labeldata2

    def __len__(self):
        return 14000


class EvalDataset1_sip_both(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset1_sip_both, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 58)) + [59]

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)

        self.df = self.df_A.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = ['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)

        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['sip'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values

    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[8000:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[8000:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df1[8000:, 55:56][idx]) - 1.

        return testdata1, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df1[8000:, :])
        return length


class EvalDataset2_sip_both(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset2_sip_both, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 58)) + [59]

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)

        self.df = self.df_A.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = []#['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)

        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['sip'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['sip'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values

        self.df2_test = self.np_df2[135:, :]

        for i in range(90):
            self.df2_test = np.concatenate((self.df2_test, self.np_df2[135:, :]))


    def __getitem__(self, idx):

        testdata2 = torch.FloatTensor(self.np_df2[135:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df2[135:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df2[135:, 55:56][idx]) - 1.

        return testdata2, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df2[135:, :])
        return length


class TrainDataset_nec_i(Dataset):
    def __init__(self, knnfile):
        super(TrainDataset_nec_i, self).__init__()

        self.knnfile = knnfile

        self.data = list(range(1, 58)) + [59]
        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)
        self.df = self.df_A.drop(['svamal'], axis=1)
        self.data_ = self.df.copy(deep = True)

        self.mode_feature = []#['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)
        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['nec'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['nec'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values

        self.df2_train = self.np_df2[:700, :]

        for i in range(10):
            self.df2_train = np.concatenate((self.df2_train, self.np_df2[:700, :]))

        # print(self.df.info())

    def __getitem__(self, idx):

        t = np.concatenate((self.np_df1[1000:8000, :54], self.df2_train[:7000, :54]), axis=0)
        l1 = np.concatenate((self.np_df1[1000:8000, 54], self.df2_train[:7000, 54]), axis=0)
        l2 = np.concatenate((self.np_df1[1000:8000, 55], self.df2_train[:7000, 55]), axis=0)

        traindata = torch.as_tensor(t[idx]).float()
        labeldata1 = torch.as_tensor(l1[idx]).float() - 1.
        labeldata2 = torch.as_tensor(l2[idx]).float() - 1.

        return traindata, labeldata1, labeldata2

    def __len__(self):
        return 14000


class EvalDataset1_nec_i(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset1_nec_i, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 58)) + [59]

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)

        self.df = self.df_A.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = []#['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)

        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['nec'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values

    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[8000:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[8000:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df1[8000:, 55:56][idx]) - 1.

        return testdata1, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df1[8000:, :])
        return length


class EvalDataset2_nec_i(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset2_nec_i, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 58)) + [59]

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)

        self.df = self.df_A.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = []#['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)

        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['nec'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['nec'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values

        self.df2_test = self.np_df2[700:, :]

        for i in range(90):
            self.df2_test = np.concatenate((self.df2_test, self.np_df2[700:, :]))

    def __getitem__(self, idx):

        testdata2 = torch.FloatTensor(self.np_df2[700:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df2[700:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df2[700:, 55:56][idx]) - 1.

        return testdata2, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df2[700:, :])
        return length


class TrainDataset_ip_both(Dataset):
    def __init__(self, knnfile):
        super(TrainDataset_ip_both, self).__init__()

        self.knnfile = knnfile

        self.data = list(range(1, 58)) + [60]
        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)
        self.df = self.df_A.drop(['svamal'], axis=1)
        self.data_ = self.df.copy(deep = True)

        self.mode_feature = ['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)
        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['ip'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['ip'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values

        self.df2_train = self.np_df2[:540, :]

        for i in range(13):
            self.df2_train = np.concatenate((self.df2_train, self.np_df2[:540, :]))

        print(self.df.info())

    def __getitem__(self, idx):

        t = np.concatenate((self.np_df1[1000:8000, :54], self.df2_train[:7000, :54]), axis=0)
        l1 = np.concatenate((self.np_df1[1000:8000, 54], self.df2_train[:7000, 54]), axis=0)
        l2 = np.concatenate((self.np_df1[1000:8000, 55], self.df2_train[:7000, 55]), axis=0)

        traindata = torch.as_tensor(t[idx]).float()
        labeldata1 = torch.as_tensor(l1[idx]).float() - 1.
        labeldata2 = torch.as_tensor(l2[idx]).float() - 1.

        return traindata, labeldata1, labeldata2

    def __len__(self):
        return 14000


class EvalDataset1_ip_both(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset1_ip_both, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 58)) + [60]

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)

        self.df = self.df_A.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = ['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)

        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['ip'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values

    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[8000:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[8000:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df1[8000:, 55:56][idx]) - 1.

        return testdata1, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df1[8000:, :])
        return length


class EvalDataset2_ip_both(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset2_ip_both, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 58)) + [60]

        self.df_A = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data)

        self.df = self.df_A.drop(['svamal'], axis=1)

        self.data_ = self.df.copy(deep = True)

        self.mode_feature = ['apgs1', 'apgs5', 'medu', 'fedu']
        self.mean_feature = ['bhei', 'bhead', 'btem', 'bbph']

        for self.header in self.data_.keys():
            if self.header in self.mode_feature:
                self.data_[self.header].fillna(self.data_[self.header].mode()[0], inplace = True)
            elif self.header in self.mean_feature:
                self.data_[self.header].fillna(self.data_[self.header].mean(), inplace = True)
            else:
                self.data_[self.header].fillna(self.data_[self.header].median(), inplace = True)

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm = normalize1(self.df_train)

        self.df = self.df_train_norm.join(self.df_test)

        self.ntet1 = self.df['ip'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['ip'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values

        self.df2_test = self.np_df2[540:, :]

        # for i in range(90):
        #     self.df2_test = np.concatenate((self.df2_test, self.np_df2[540:, :]))

    def __getitem__(self, idx):

        testdata2 = torch.FloatTensor(self.np_df2[540:, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df2[540:, 54:55][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df2[540:, 55:56][idx]) - 1.

        return testdata2, labeldata1, labeldata2

    def __len__(self):
        length = len(self.np_df2[540:, :])
        return length
