import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve
import matplotlib.pyplot as plt

from model import *
from data_loader8 import *
from earlystop import EarlyStopping

def roc_curve_plot(y_test, pred_proba_c1, epoch):
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)

    plt.clf()

    plt.plot(fprs, tprs, label='ROC')
    plt.plot([0,1], [0,1], 'k--', label='Random')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.savefig('/home/daehyun/KNN/semi_final/roc/{}.png'.format(epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="/home/daehyun/KNN/data_v2.csv")
    parser.add_argument('--outputs-dir', type=str, default="/home/daehyun/KNN/semi_final/weight")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir)
    out_path = args.outputs_dir

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = KNN().to(device)
    learning_rate = args.lr

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=7, verbose=True)

    train_dataset = TrainDataset_ntet(args.data)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=False,
                                  drop_last=True)

    eval_dataset1 = EvalDataset1_ntet(args.data)
    eval_dataloader1 = DataLoader(dataset=eval_dataset1, batch_size=1)
    eval_dataset2 = EvalDataset2_ntet(args.data)
    eval_dataloader2 = DataLoader(dataset=eval_dataset2, batch_size=1)

    min_loss = 10
    for epoch in range(args.num_epochs):

        model.train()

        loss_sum = 0
        cnt = 0

        for data in train_dataloader:
            
            inputs, labels = data
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            preds = model(inputs)

            loss = criterion(preds, labels)
            loss_sum += loss
            cnt += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        accuracy1, accuracy2 = 0, 0
        testloss1, testloss2 = 0, 0
        n1, n2 = 0, 0

        label = []
        predict1 = []
        predict2 = []

        for data in eval_dataloader1:

            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs)
            
            label += labels.tolist()
            predict1 += preds.tolist()

            testloss1 += criterion(preds, labels)

            if preds > 0.5:
                preds = 1
            else:
                preds = 0
            
            if preds == labels:
                accuracy1 += 1
            else:
                accuracy1 += 0

            n1 += 1

            predict2 += [preds]


        for data in eval_dataloader2:

            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs)

            label += labels.tolist()
            predict1 += preds.tolist()

            testloss2 += criterion(preds, labels)

            if preds > 0.5:
                preds = 1
            else:
                preds = 0
            
            if preds == labels:
                accuracy2 += 1
            else:
                accuracy2 += 0

            n2 += 1
            predict2 += [preds]


        if epoch == 0:
            print(n1, n2)

        auc_acc = roc_auc_score(label, predict1)

        testloss = (testloss1/n1) + (testloss2/n2)
        accur = (accuracy1/n1 + accuracy2/n2) * 0.5

        print("Test Loss : {}".format(testloss))
        print("EPOCH : {0:3d}  AUCacc : {1:0.4f}  Test : {2:0.4f}  Test1 : {3:0.4f}  Test2 : {4:0.4f}".format(epoch, auc_acc, accur, accuracy1/n1, accuracy2/n2))

        if epoch > 2:
            early_stopping(testloss, model)
            
            if min_loss > testloss:
                min_loss = testloss
                roc_curve_plot(label, predict1, epoch)

        if early_stopping.early_stop:
            print('stop!')
            print('minimun loss is ', min_loss)
            break



