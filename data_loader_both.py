import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def normalize(input):
    return (input - input.mean()) / input.std()


def normalize1(x):
    return (x - x.min()) / (x.max() - x.min())


class TrainDataset(Dataset):
    def __init__(self, knnfile):
        super(TrainDataset, self).__init__()

        self.knnfile = knnfile

        self.data = list(range(1, 57)) + [57]
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

        self.df_train1 = self.data_.loc[:, :'strdu']
        self.df_train2 = self.data_.loc[:, 'indopda':'ibupda']
        self.df_train3 = self.data_.loc[:, 'lbp':'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm1 = normalize1(self.df_train1)
        self.df_train_norm2 = normalize1(self.df_train2)
        self.df_train_norm3 = normalize1(self.df_train3)

        self.df_norm1 = self.df_train_norm1.join(self.df_train_norm3)
        self.df_norm2 = self.df_norm1.join(self.df_train_norm2)

        self.df = self.df_norm2.join(self.df_test)

        self.ntet1 = self.df['ntet'] == 1
        self.df1 = self.df[self.ntet1]
        self.a1 = self.df1['pdad'] == 1
        self.a2 = self.df1['pdad'] == 2
        self.df11 = self.df1[self.a1]
        self.df12 = self.df1[self.a2]
        self.np_df11 = self.df11.values
        self.np_df12 = self.df12.values

        self.ntet2 = self.df['ntet'] == 2
        self.df2 = self.df[self.ntet2]
        self.b1 = self.df2['pdad'] == 1
        self.b2 = self.df2['pdad'] == 2
        self.df21 = self.df2[self.b1]
        self.df22 = self.df2[self.b2]
        self.np_df21 = self.df21.values
        self.np_df22 = self.df22.values

        print(len(self.df11), len(self.df12), len(self.df21), len(self.df22))

        self.df12_train = np.concatenate((self.np_df12[:1500, :], self.np_df12[:1500, :]))

        self.df21_train = self.np_df21[:350, :]
        for i in range(10):
            self.df21_train = np.concatenate((self.df21_train, self.np_df21[:350, :]))


        self.df22_train = self.np_df22[:230, :]
        for i in range(20):
            self.df22_train = np.concatenate((self.df22_train, self.np_df22[:230, :]))


    def __getitem__(self, idx):

        t = np.concatenate((self.np_df11[:3000, :54], self.df12_train[:3000, :54], self.df21_train[:3000, :54], self.df22_train[:3000, :54]), axis=0)
        l_p = np.concatenate((self.np_df11[:3000, 52], self.df12_train[:3000, 52], self.df21_train[:3000, 52], self.df22_train[:3000, 52]), axis=0)
        l_n = np.concatenate((self.np_df11[:3000, 54], self.df12_train[:3000, 54], self.df21_train[:3000, 54], self.df22_train[:3000, 54]), axis=0)

        traindata = torch.as_tensor(t[idx]).float()
        labeldata1 = torch.as_tensor(l_p[idx]).float() - 1.
        labeldata2 = torch.as_tensor(l_n[idx]).float() - 1.

        return traindata, labeldata1, labeldata2

    def __len__(self):
        return 12000


class EvalDataset1(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset1, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 57)) + [57]

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

        self.df_train1 = self.data_.loc[:, :'strdu']
        self.df_train2 = self.data_.loc[:, 'indopda':'ibupda']
        self.df_train3 = self.data_.loc[:, 'lbp':'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm1 = normalize1(self.df_train1)
        self.df_train_norm2 = normalize1(self.df_train2)
        self.df_train_norm3 = normalize1(self.df_train3)

        self.df_norm1 = self.df_train_norm1.join(self.df_train_norm3)
        self.df_norm2 = self.df_norm1.join(self.df_train_norm2)

        self.df = self.df_norm2.join(self.df_test)

        self.ntet1 = self.df['ntet'] == 1
        self.df1 = self.df[self.ntet1]
        self.a1 = self.df1['pdad'] == 1
        self.a2 = self.df1['pdad'] == 2
        self.df11 = self.df1[self.a1]
        self.df12 = self.df1[self.a2]
        self.np_df11 = self.df11.values
        self.np_df12 = self.df12.values

        self.ntet2 = self.df['ntet'] == 2
        self.df2 = self.df[self.ntet2]
        self.b1 = self.df2['pdad'] == 1
        self.b2 = self.df2['pdad'] == 2
        self.df21 = self.df2[self.b1]
        self.df22 = self.df2[self.b2]
        self.np_df21 = self.df21.values
        self.np_df22 = self.df22.values

        self.df12_test = np.concatenate((self.np_df12[1500:, :], self.np_df12[1500:, :]))

        self.df21_test = self.np_df21[350:, :]
        for i in range(10):
            self.df21_test = np.concatenate((self.df21_test, self.np_df21[350:, :]))


        self.df22_test = self.np_df22[230:, :]
        for i in range(60):
            self.df22_test = np.concatenate((self.df22_test, self.np_df22[230:, :]))


    def __getitem__(self, idx):

        t = np.concatenate((self.np_df11[3500:6500, :54], self.df12_test[:3000, :54]), axis=0)
        l_p = np.concatenate((self.np_df11[3500:6500, 52], self.df12_test[:3000, 52]), axis=0)
        l_n = np.concatenate((self.np_df11[3500:6500, 54], self.df12_test[:3000, 54]), axis=0)

        testdata1 = torch.as_tensor(t[idx]).float()
        labeldata1 = torch.as_tensor(l_p[idx]).float() - 1.
        labeldata2 = torch.as_tensor(l_n[idx]).float() - 1.

        return testdata1, labeldata1, labeldata2

    def __len__(self):
        return 6000


class EvalDataset2(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset2, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 57)) + [57]

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

        self.df_train1 = self.data_.loc[:, :'strdu']
        self.df_train2 = self.data_.loc[:, 'indopda':'ibupda']
        self.df_train3 = self.data_.loc[:, 'lbp':'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]

        self.df_train_norm1 = normalize1(self.df_train1)
        self.df_train_norm2 = normalize1(self.df_train2)
        self.df_train_norm3 = normalize1(self.df_train3)

        self.df_norm1 = self.df_train_norm1.join(self.df_train_norm3)
        self.df_norm2 = self.df_norm1.join(self.df_train_norm2)

        self.df = self.df_norm2.join(self.df_test)

        self.ntet1 = self.df['ntet'] == 1
        self.df1 = self.df[self.ntet1]
        self.a1 = self.df1['pdad'] == 1
        self.a2 = self.df1['pdad'] == 2
        self.df11 = self.df1[self.a1]
        self.df12 = self.df1[self.a2]
        self.np_df11 = self.df11.values
        self.np_df12 = self.df12.values

        self.ntet2 = self.df['ntet'] == 2
        self.df2 = self.df[self.ntet2]
        self.b1 = self.df2['pdad'] == 1
        self.b2 = self.df2['pdad'] == 2
        self.df21 = self.df2[self.b1]
        self.df22 = self.df2[self.b2]
        self.np_df21 = self.df21.values
        self.np_df22 = self.df22.values

        self.df12_test = np.concatenate((self.np_df12[1500:, :], self.np_df12[1500:, :]))

        self.df21_test = self.np_df21[350:, :]
        for i in range(50):
            self.df21_test = np.concatenate((self.df21_test, self.np_df21[350:, :]))


        self.df22_test = self.np_df22[230:, :]
        for i in range(60):
            self.df22_test = np.concatenate((self.df22_test, self.np_df22[230:, :]))


    def __getitem__(self, idx):
        
        t = np.concatenate((self.df21_test[:3000, :54], self.df22_test[:3000, :54]), axis=0)
        l_p = np.concatenate((self.df21_test[:3000, 52], self.df22_test[:3000, 52]), axis=0)
        l_n = np.concatenate((self.df21_test[:3000, 54], self.df22_test[:3000, 54]), axis=0)

        testdata2 = torch.as_tensor(t[idx]).float()
        labeldata1 = torch.as_tensor(l_p[idx]).float() - 1.
        labeldata2 = torch.as_tensor(l_n[idx]).float() - 1.

        return testdata2, labeldata1, labeldata2

    def __len__(self):
        return 6000
