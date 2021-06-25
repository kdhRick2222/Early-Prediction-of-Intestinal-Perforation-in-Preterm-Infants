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
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['ntet'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values

        self.df2_train = self.np_df2[:600, :]

        for i in range(5):
            self.df2_train = np.concatenate((self.df2_train, self.np_df2[:600, :]))

        # self.data1 = list(range(1, 56))
        # self.df_B = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data1)
        # self.df_b = self.df_B.drop(['indopda', 'ibupda', 'svamal'], axis=1)
        # #self.df_b = self.df_B.drop(['svamal'], axis=1)
        # self.data_b = self.df_b.copy(deep = True)

        # for self.header in self.data_b.keys():
        #     if self.header in self.mode_feature:
        #         self.data_b[self.header].fillna(self.data_b[self.header].mode()[0], inplace = True)
        #     elif self.header in self.mean_feature:
        #         self.data_b[self.header].fillna(self.data_b[self.header].mean(), inplace = True)
        #     else:
        #         self.data_b[self.header].fillna(self.data_b[self.header].median(), inplace = True)

        # self.pdad_train = self.data_b.loc[:, :'eythtran']
        # self.pdad_test = self.data_b.loc[:, 'pdad']

        # self.pdad_train_norm = normalize1(self.pdad_train)

        # self.pdad = self.pdad_train_norm.join(self.pdad_test)

        # self.pdad1 = self.pdad['pdad'] == 1
        # self.df_pdad1 = self.pdad[self.pdad1]
        # self.np_pdad1 = self.df_pdad1.values
        # self.pdad2 = self.pdad['pdad'] == 2
        # self.df_pdad2 = self.pdad[self.pdad2]
        # self.np_pdad2 = self.df_pdad2.values

        print(self.df.info())
        # print(self.pdad.info())

    def __getitem__(self, idx):

        t = np.concatenate((self.np_df1[:3000, :54], self.df2_train[:3000, :54]), axis=0)
        l_p = np.concatenate((self.np_df1[:3000, 52], self.df2_train[:3000, 52]), axis=0)
        l_n = np.concatenate((self.np_df1[:3000, 54], self.df2_train[:3000, 54]), axis=0)

        traindata = torch.as_tensor(t[idx]).float()
        labeldata1 = torch.as_tensor(l_p[idx]).float() - 1.
        labeldata2 = torch.as_tensor(l_n[idx]).float() - 1.

        return traindata, labeldata1, labeldata2

    def __len__(self):
        return 6000


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
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['ntet'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values


        # self.data1 = list(range(1, 56))
        # self.df_B = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data1)
        # self.df_b = self.df_B.drop(['indopda', 'ibupda', 'svamal'], axis=1)
        # #self.df_b = self.df_B.drop(['svamal'], axis=1)
        # self.data_b = self.df_b.copy(deep = True)

        # for self.header in self.data_b.keys():
        #     if self.header in self.mode_feature:
        #         self.data_b[self.header].fillna(self.data_b[self.header].mode()[0], inplace = True)
        #     elif self.header in self.mean_feature:
        #         self.data_b[self.header].fillna(self.data_b[self.header].mean(), inplace = True)
        #     else:
        #         self.data_b[self.header].fillna(self.data_b[self.header].median(), inplace = True)

        # self.pdad_train = self.data_b.loc[:, :'eythtran']
        # self.pdad_test = self.data_b.loc[:, 'pdad']

        # self.pdad_train_norm = normalize1(self.pdad_train)

        # self.pdad = self.pdad_train_norm.join(self.pdad_test)

        # self.pdad1 = self.pdad['pdad'] == 1
        # self.df_pdad1 = self.pdad[self.pdad1]
        # self.np_pdad1 = self.df_pdad1.values
        # self.pdad2 = self.pdad['pdad'] == 2
        # self.df_pdad2 = self.pdad[self.pdad2]
        # self.np_pdad2 = self.df_pdad2.values


    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[3000:6000, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[3000:6000, 52:53][idx]) - 1.
        labeldata2 = torch.FloatTensor(self.np_df1[3000:6000, 54:][idx]) - 1.

        return testdata1, labeldata1, labeldata2

    def __len__(self):
        return 3000


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
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['ntet'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values

        self.df2_test = self.np_df2[600:, :]

        for i in range(90):
            self.df2_test = np.concatenate((self.df2_test, self.np_df2[600:, :]))

        
        # self.data1 = list(range(1, 56))
        # self.df_B = pd.read_csv(self.knnfile, index_col = 0, usecols = self.data1)
        # self.df_b = self.df_B.drop(['indopda', 'ibupda', 'svamal'], axis=1)
        # #self.df_b = self.df_B.drop(['svamal'], axis=1)
        # self.data_b = self.df_b.copy(deep = True)

        # for self.header in self.data_b.keys():
        #     if self.header in self.mode_feature:
        #         self.data_b[self.header].fillna(self.data_b[self.header].mode()[0], inplace = True)
        #     elif self.header in self.mean_feature:
        #         self.data_b[self.header].fillna(self.data_b[self.header].mean(), inplace = True)
        #     else:
        #         self.data_b[self.header].fillna(self.data_b[self.header].median(), inplace = True)

        # self.pdad_train = self.data_b.loc[:, :'eythtran']
        # self.pdad_test = self.data_b.loc[:, 'pdad']

        # self.pdad_train_norm = normalize1(self.pdad_train)

        # self.pdad = self.pdad_train_norm.join(self.pdad_test)

        # self.pdad1 = self.pdad['pdad'] == 1
        # self.df_pdad1 = self.pdad[self.pdad1]
        # self.np_pdad1 = self.df_pdad1.values
        # self.pdad2 = self.pdad['pdad'] == 2
        # self.df_pdad2 = self.pdad[self.pdad2]
        # self.np_pdad2 = self.df_pdad2.values

        # self.df2_pdad = self.np_pdad2[3000:, :]

        # for i in range(20):
        #     self.df2_pdad = np.concatenate((self.df2_pdad, self.np_pdad2[3000:, :]))

    def __getitem__(self, idx):
        testdata2 = torch.FloatTensor(self.df2_test[:3000, :54][idx])
        labeldata2 = torch.FloatTensor(self.df2_test[:3000, 52:53][idx]) - 1.
        labeldata3 = torch.FloatTensor(self.df2_test[:3000, 54:][idx]) - 1.

        return testdata2, labeldata2, labeldata3

    def __len__(self):
        return 3000


