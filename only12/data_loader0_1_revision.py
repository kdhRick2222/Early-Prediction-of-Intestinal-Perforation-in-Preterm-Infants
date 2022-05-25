import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize1(x):
    return (x - x.min()) / (x.max() - x.min())


class TrainDataset_nec(Dataset):
    def __init__(self, knnfile):
        super(TrainDataset_nec, self).__init__()
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

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]
        # self.df_train = self.data_.loc[:, :'acl']
        # self.df_test = self.data_.loc[:, 'nec':]

        self.df_train_norm = normalize1(self.df_train)

        self.df = self.df_train_norm.join(self.df_test)

        #self.df['btem']=0

        self.ntet1 = self.df['nec'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['nec'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values

        self.df2_train = np.concatenate((self.np_df2[:700, :], self.np_df2[772:,:]))
        self.df2_train2 = np.concatenate((self.np_df2[:700, :], self.np_df2[772:,:]))

        for i in range(14):
            self.df2_train = np.concatenate((self.df2_train, self.df2_train2))

        print(self.df.info())

    def __getitem__(self, idx):

        t = np.concatenate((self.np_df1[:5000, :54], self.df2_train[:5000, :54]), axis=0)
        l = np.concatenate((self.np_df1[:5000, 54:], self.df2_train[:5000, 54:]), axis=0)

        traindata = torch.as_tensor(t[idx]).float()
        labeldata = torch.as_tensor(l[idx]).float() - 1.

        return traindata, labeldata

    def __len__(self):
        return len(np.concatenate((self.np_df1[:5000, 54:], self.df2_train[:5000, 54:]), axis=0))


class EvalDataset1_nec(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset1_nec, self).__init__()
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

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]
        # self.df_train = self.data_.loc[:, :'acl']
        # self.df_test = self.data_.loc[:, 'nec':]

        self.df_train_norm = normalize1(self.df_train)

        self.df = self.df_train_norm.join(self.df_test)

        #self.df['btem']=0

        self.ntet1 = self.df['nec'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['nec'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values


    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[8000:9618, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[8000:9618, 54:][idx]) - 1.
        return testdata1, labeldata1

    def __len__(self):
        length = len(self.np_df1[8000:9618, :])
        return length


class EvalDataset2_nec(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset2_nec, self).__init__()
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

        self.df_train = self.data_.loc[:, :'eythtran']
        self.df_test = self.data_.loc[:, 'pdad':]
        # self.df_train = self.data_.loc[:, :'acl']
        # self.df_test = self.data_.loc[:, 'nec':]

        self.df_train_norm = normalize1(self.df_train)

        self.df = self.df_train_norm.join(self.df_test)

        #self.df['btem']=0

        self.ntet1 = self.df['nec'] == 1
        self.df1 = self.df[self.ntet1]
        self.np_df1 = self.df1.values
        self.ntet2 = self.df['nec'] == 2
        self.df2 = self.df[self.ntet2]
        self.np_df2 = self.df2.values

        self.df2_test = self.np_df2[770:, :]

        # for i in range(23):
        #     self.df2_test = np.concatenate((self.df2_test, self.np_df2[700:, :]))

    def __getitem__(self, idx):
        testdata2 = torch.FloatTensor(self.np_df2[700:772, :54][idx])
        labeldata2 = torch.FloatTensor(self.np_df2[700:772, 54:][idx]) - 1.
        # testdata2 = torch.FloatTensor(self.df2_test[:1618, :54][idx])
        # labeldata2 = torch.FloatTensor(self.df2_test[:1618, 54:][idx]) - 1.
        return testdata2, labeldata2

    def __len__(self):
        # length = len(self.df2_test[:1618, :])
        length = len(self.np_df2[700:772, :])
        return length


class TrainDataset_necip(Dataset):
    def __init__(self, knnfile):
        super(TrainDataset_necip, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 57)) + [58]

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

        #self.df['acl']=0

        self.ntety1 = self.df['necip'] == 1
        self.df1 = self.df[self.ntety1]
        self.np_df1 = self.df1.values
        self.ntety2 = self.df['necip'] == 2
        self.df2 = self.df[self.ntety2]
        self.np_df2 = self.df2.values

        self.df2_train = np.concatenate((self.np_df2[:380, :], self.np_df2[428:, :]))
        self.df2_train2 = np.concatenate((self.np_df2[:380, :], self.np_df2[428:, :]))

        for i in range(24):
            self.df2_train = np.concatenate((self.df2_train, self.df2_train2))

        print(self.df.info())

    def __getitem__(self, idx):

        t = np.concatenate((self.np_df1[:4500, :54], self.df2_train[:4500, :54]), axis=0)
        l = np.concatenate((self.np_df1[:4500, 54:], self.df2_train[:4500, 54:]), axis=0)

        traindata = torch.as_tensor(t[idx]).float()
        labeldata = torch.as_tensor(l[idx]).float() - 1.

        return traindata, labeldata

    def __len__(self):
        return len(np.concatenate((self.np_df1[:4500, 54:], self.df2_train[:4500, 54:]), axis=0))


class EvalDataset1_necip(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset1_necip, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 57)) + [58]

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

        #self.df['acl']=0

        self.ntety1 = self.df['necip'] == 1
        self.df1 = self.df[self.ntety1]
        self.np_df1 = self.df1.values
        self.ntety2 = self.df['necip'] == 2
        self.df2 = self.df[self.ntety2]
        self.np_df2 = self.df2.values


    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[8000:9962, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[8000:9962, 54:][idx]) - 1.
        return testdata1, labeldata1

    def __len__(self):
        length = len(self.np_df1[8000:9962, :54])
        return length


class EvalDataset2_necip(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset2_necip, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 57)) + [58]

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

        #self.df['acl']=0

        self.ntety1 = self.df['necip'] == 1
        self.df1 = self.df[self.ntety1]
        self.np_df1 = self.df1.values
        self.ntety2 = self.df['necip'] == 2
        self.df2 = self.df[self.ntety2]
        self.np_df2 = self.df2.values

        self.df2_test = self.np_df2[380:428, :]

        # for i in range(20):
        #     self.df2_test = np.concatenate((self.df2_test, self.np_df2[430:, :]))

    def __getitem__(self, idx):
        testdata2 = torch.FloatTensor(self.np_df2[380:428, :54][idx])
        labeldata2 = torch.FloatTensor(self.np_df2[380:428, 54:][idx]) - 1.
        return testdata2, labeldata2

    def __len__(self):
        length = len(self.np_df2[380:428, 54:])
        return length


class TrainDataset_sip(Dataset):
    def __init__(self, knnfile):
        super(TrainDataset_sip, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 57)) + [59]

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

        #self.df['acl']=0

        self.iperr1 = self.df['sip'] == 1
        self.df1 = self.df[self.iperr1]
        self.np_df1 = self.df1.values
        self.iperr2 = self.df['sip'] == 2
        self.df2 = self.df[self.iperr2]
        self.np_df2 = self.df2.values

        # self.df2_train = np.concatenate((self.np_df2[:135, :], self.np_df2[165:, :]))
        # self.df2_train2 = np.concatenate((self.np_df2[:135, :], self.np_df2[165:, :]))
        self.df2_train = self.np_df2[:170, :]

        for i in range(67):
            self.df2_train = np.concatenate((self.df2_train, self.np_df2[:170,:]))

        # print(self.df.info())

    def __getitem__(self, idx):

        t = np.concatenate((self.np_df1[:4500, :54], self.df2_train[:4500, :54]), axis=0)
        l = np.concatenate((self.np_df1[:4500, 54:], self.df2_train[:4500, 54:]), axis=0)

        traindata = torch.as_tensor(t[idx]).float()
        labeldata = torch.as_tensor(l[idx]).float() - 1.

        return traindata, labeldata

    def __len__(self):
        return len(np.concatenate((self.np_df1[:4500, :], self.df2_train[:4500, :]), axis=0))


class EvalDataset1_sip(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset1_sip, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 57)) + [59]

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

        #self.df['acl']=0

        self.iperr1 = self.df['sip'] == 1
        self.df1 = self.df[self.iperr1]
        self.np_df1 = self.df1.values
        self.iperr2 = self.df['sip'] == 2
        self.df2 = self.df[self.iperr2]
        self.np_df2 = self.df2.values


    def __getitem__(self, idx):
        testdata1 = torch.FloatTensor(self.np_df1[8000:10000, :54][idx])
        labeldata1 = torch.FloatTensor(self.np_df1[8000:10000, 54:][idx]) - 1.
        return testdata1, labeldata1

    def __len__(self):
        length = len(self.np_df1[8000:10000, :])
        return length


class EvalDataset2_sip(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset2_sip, self).__init__()
        self.knnfile = knnfile
        self.data = list(range(1, 57)) + [59]

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

        #self.df['acl']=0

        self.iperr1 = self.df['sip'] == 1
        self.df1 = self.df[self.iperr1]
        self.np_df1 = self.df1.values
        self.iperr2 = self.df['sip'] == 2
        self.df2 = self.df[self.iperr2]
        self.np_df2 = self.df2.values

        self.df2_test = self.np_df2[135:166, :]

        # for i in range(60):
            # self.df2_test = np.concatenate((self.df2_test, self.np_df2[150:, :]))

    def __getitem__(self, idx):
        testdata2 = torch.FloatTensor(self.np_df2[170:, :54][idx])
        labeldata2 = torch.FloatTensor(self.np_df2[170:, 54:][idx]) - 1.
        return testdata2, labeldata2

    def __len__(self):
        length = len(self.np_df2[170:, :])
        return length


