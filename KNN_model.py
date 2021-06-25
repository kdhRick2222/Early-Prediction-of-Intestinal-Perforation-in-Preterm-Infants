import torch
from torch import nn
import math


class pdad_ntet(nn.Module):
    def __init__(self):
        super(pdad_ntet, self).__init__()

        self.ntet1 = nn.Linear(54, 32)
        self.ntet2 = nn.Linear(32, 16)
        self.ntet3 = nn.Linear(16, 8)
        self.ntet4 = nn.Linear(16, 8)
        self.ntet5 = nn.Linear(8, 4)
        self.ntet6 = nn.Linear(4, 1)

        self.pdad1 = nn.Linear(50, 64)
        self.pdad2 = nn.Linear(64, 32) 
        self.pdad3 = nn.Linear(32, 16)
        self.pdad4 = nn.Linear(16, 8)
        self.pdad5 = nn.Linear(8, 4)
        self.pdad6 = nn.Linear(4, 1)

        self.bn64 = nn.BatchNorm1d(64)
        self.bn32 = nn.BatchNorm1d(32)
        self.bn16 = nn.BatchNorm1d(16)
        self.bn8 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):

        x = self.bn64(self.dropout(self.relu(self.pdad1(x))))
        x = self.bn32(self.dropout(self.relu(self.pdad2(x))))
        x = self.bn16(self.dropout(self.relu(self.pdad3(x))))
        x1 = self.bn8(self.dropout(self.relu(self.pdad4(x))))
        x = self.bn4(self.dropout(self.relu(self.pdad5(x1))))
        x = self.sigmoid(self.pdad6(x))

        y = self.bn32(self.dropout(self.relu(self.ntet1(y))))
        y = self.bn16(self.dropout(self.relu(self.ntet2(y))))
        y = self.bn8(self.dropout(self.relu(self.ntet3(y))))
        y1 = torch.cat((y, x1), dim=1)
        y = self.bn8(self.dropout(self.relu(self.ntet4(y1))))
        y = self.bn4(self.dropout(self.relu(self.ntet5(y))))
        y = self.sigmoid(self.ntet6(y))

        return x, y

