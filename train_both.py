import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

from KNN_model import pdad_ntet
from data_loader_both import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="/home/daehyun/KNN/data_v2.csv")
    parser.add_argument('--outputs-dir', type=str, default="/home/daehyun/KNN/weight")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir)
    out_path = args.outputs_dir

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(123)

    model = pdad_ntet().to(device)
    learning_rate = args.lr

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TrainDataset(args.data)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=False,
                                  drop_last=True)

    eval_dataset1 = EvalDataset1(args.data)
    eval_dataloader1 = DataLoader(dataset=eval_dataset1, batch_size=1)
    eval_dataset2 = EvalDataset2(args.data)
    eval_dataloader2 = DataLoader(dataset=eval_dataset2, batch_size=1)


    for epoch in range(args.num_epochs):

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loss_sum1 = 0
        loss_sum2 = 0

        for data in train_dataloader:
            
            inputs, labels1_, labels2_ = data
            
            inputs = inputs.to(device)
            labels1_ = labels1_.to(device)
            labels2_ = labels2_.to(device)

            labels1 = labels1_.unsqueeze(1)
            labels2 = labels2_.unsqueeze(1)
            
            pdad, ntet = model(inputs[:, :50], inputs)

            loss1 = criterion(pdad, labels1)
            loss2 = criterion(ntet, labels2)
            loss = loss1 + loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum1 += loss1
            loss_sum2 += loss2
 
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'train_both_{}.pth'.format(epoch)))

        model.eval()

        pdad1_acc, pdad2_acc, ntet1_acc, ntet2_acc = 0, 0, 0, 0
        pdad_loss1, pdad_loss2, ntet_loss1, ntet_loss2 = 0, 0, 0, 0

        n11, n12, n21, n22 = 0, 0, 0, 0

        for data in eval_dataloader1:

            inputs, labels1_, labels2_ = data

            inputs = inputs.to(device)
            labels1_ = labels1_.to(device)
            labels2_ = labels2_.to(device)

            with torch.no_grad():
                pdad, ntet = model(inputs[:, :50], inputs)

            labels1 = labels1_.unsqueeze(1)
            labels2 = labels2_.unsqueeze(1)
            
            pdad_loss1 += criterion(pdad, labels1)
            ntet_loss1 += criterion(ntet, labels2)

            if pdad > 0.5:
                pdad = 1
            else:
                pdad = 0
            
            if labels1 == 0:
                n11 += 1
                if pdad == 0:
                    pdad1_acc += 1
                else:
                    pdad1_acc += 0
            else:
                n12 += 1
                if pdad == 1:
                    pdad2_acc += 1
                else:
                    pdad2_acc += 0
            
            if ntet > 0.5:
                ntet = 1
            else:
                ntet = 0
            
            if ntet == 0:
                ntet1_acc += 1
            else:
                ntet1_acc += 0


        for data in eval_dataloader2:

            inputs1, labels1_, labels2_ = data

            inputs = inputs.to(device)
            labels1_ = labels1_.to(device)
            labels2_ = labels2_.to(device)

            with torch.no_grad():
                pdad, ntet = model(inputs[:, :50], inputs)

            labels1 = labels1_.unsqueeze(1)
            labels2 = labels2_.unsqueeze(1)
            
            pdad_loss2 += criterion(pdad, labels1)
            ntet_loss2 += criterion(ntet, labels2)

            if pdad > 0.5:
                pdad = 1
            else:
                pdad = 0

            if labels1 == 0:
                n21 += 1
                if pdad == 0:
                    pdad1_acc += 1
                else:
                    pdad1_acc += 0
            else:
                n22 += 1
                if pdad == 1:
                    pdad2_acc += 1
                else:
                    pdad2_acc += 0
            
            if ntet > 0.5:
                ntet = 1
            else:
                ntet = 0
            
            if ntet == 1:
                ntet2_acc += 1
            else:
                ntet2_acc += 0
        
        n1 = n11 + n12
        n2 = n21 + n22
        
        pdad_loss = (pdad_loss1 + pdad_loss2) / (n1 + n2)
        ntet_loss = (ntet_loss1 + ntet_loss2) /(n1 + n2)
        ntet_acc = (ntet1_acc/n1 + ntet2_acc/n2) * 0.5
        pdad_acc = (pdad1_acc / (n11 + n21) + pdad2_acc / (n12 + n22)) * 0.5
        pdad1_ = pdad1_acc / (n11 + n21)
        pdad2_ = pdad2_acc / (n12 + n22)

        print("EPOCH : {0:3d}  pdad_loss : {1:0.4f}  ntet_loss : {2:0.4f}  pdad : {3:0.4f}  ntet : {4:0.4f}  pdad1 : {5:0.4f}  pdad2 : {6:0.4f}  ntet1 : {7:0.4f}  ntet2 : {8:0.4f}".format(epoch, pdad_loss, ntet_loss, pdad_acc, ntet_acc, pdad1_, pdad2_, ntet1_acc/n1, ntet2_acc/n2))
        print("loss1 : {0:0.4f}  loss2 : {1:0.4f}".format(loss_sum1, loss_sum2))

