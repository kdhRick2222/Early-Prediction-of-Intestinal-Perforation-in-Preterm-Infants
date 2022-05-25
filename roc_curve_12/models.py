import torch
from torch import nn
import math

class KNN(nn.Module):
    def __init__(self):
        super(KNN, self).__init__()

        self.linear1 = nn.Linear(54, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, 4)
        self.linear6 = nn.Linear(4, 1)

        self.bn64 = nn.BatchNorm1d(64)
        self.bn32 = nn.BatchNorm1d(32)
        self.bn16 = nn.BatchNorm1d(16)
        self.bn8 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.bn64(self.dropout(self.relu(self.linear1(x))))
        x = self.bn32(self.dropout(self.relu(self.linear2(x))))
        x = self.bn16(self.dropout(self.relu(self.linear3(x))))
        x = self.bn8(self.dropout(self.relu(self.linear4(x))))
        x = self.bn4(self.dropout(self.relu(self.linear5(x))))
        x = self.sigmoid(self.linear6(x))

        return x


class use_ntet(nn.Module):
    def __init__(self):
        super(use_ntet, self).__init__()

        self.linear_a1 = nn.Linear(54, 32)
        self.linear_a2 = nn.Linear(32, 16)
        self.linear_a3 = nn.Linear(16, 4)
        self.linear_a4 = nn.Linear(4, 1)

        self.linear_b1 = nn.Linear(54, 32)
        self.linear_b2 = nn.Linear(32, 16) 
        self.linear_b3 = nn.Linear(16, 4)
        self.linear_b4 = nn.Linear(8, 4)
        self.linear_b5 = nn.Linear(4, 1)

        self.bn64 = nn.BatchNorm1d(64)
        self.bn32 = nn.BatchNorm1d(32)
        self.bn16 = nn.BatchNorm1d(16)
        self.bn8 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = x

        x = self.bn32(self.dropout(self.relu(self.linear_a1(x))))
        x = self.bn16(self.dropout(self.relu(self.linear_a2(x))))
        x = self.bn4(self.dropout(self.relu(self.linear_a3(x))))

        y = self.bn32(self.dropout(self.relu(self.linear_b1(y))))
        y = self.bn16(self.dropout(self.relu(self.linear_b2(y))))
        y = self.bn4(self.dropout(self.relu(self.linear_b3(y))))
        y = torch.cat((x, y), dim=1)
        y = self.bn4(self.dropout(self.relu(self.linear_b4(y))))
        y = self.sigmoid(self.linear_b5(y))

        x = self.sigmoid(self.linear_a4(x))

        return x, y


class SeNet(nn.Module):
    def __init__(self):
        super(SeNet, self).__init__()

        self.attn = [1 for i in range(54)]

        self.fi = nn.Parameter(torch.zeros(54), requires_grad=True)

        self.linear1 = nn.Linear(54, 32)
        self.linear2 = nn.Linear(32, 8)
        self.linear3 = nn.Linear(8, 1)

        self.bn64 = nn.BatchNorm1d(64)
        self.bn32 = nn.BatchNorm1d(32)
        self.bn16 = nn.BatchNorm1d(16)
        self.bn8 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        fi = (self.fi + 1) * 0.5
        x *= fi
        x = self.bn32(self.dropout(self.relu(self.linear1(x))))
        x = self.bn8(self.dropout(self.relu(self.linear2(x))))
        x = self.sigmoid(self.linear3(x))

        return x


class sharing_net(nn.Module):
    def __init__(self):
        super(sharing_net, self).__init__()

        self.linear_a1 = nn.Linear(54, 64)
        self.linear_a2 = nn.Linear(64, 32)
        self.linear_a3 = nn.Linear(32, 16)
        self.linear_a4 = nn.Linear(16, 8)
        self.linear_a5 = nn.Linear(8, 4)
        self.linear_a6 = nn.Linear(4, 1)

        self.linear_b1 = nn.Linear(16, 8)
        self.linear_b2 = nn.Linear(8, 4) 
        self.linear_b3 = nn.Linear(4, 1)

        self.bn64 = nn.BatchNorm1d(64)
        self.bn32 = nn.BatchNorm1d(32)
        self.bn16 = nn.BatchNorm1d(16)
        self.bn8 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.bn64(self.dropout(self.relu(self.linear_a1(x))))
        x = self.bn32(self.dropout(self.relu(self.linear_a2(x))))
        x = self.bn16(self.dropout(self.relu(self.linear_a3(x))))

        y = self.bn8(self.dropout(self.relu(self.linear_b1(x))))
        y = self.bn4(self.dropout(self.relu(self.linear_b2(y))))
        y = self.sigmoid(self.linear_b3(y))

        x = self.bn8(self.dropout(self.relu(self.linear_a4(x))))
        x = self.bn4(self.dropout(self.relu(self.linear_a5(x))))
        x = self.sigmoid(self.linear_a6(x))

        return x, y