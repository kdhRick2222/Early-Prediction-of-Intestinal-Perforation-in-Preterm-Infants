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
import numpy as np
from numpy import sqrt, argmax
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix
from sklearn.preprocessing import Binarizer

from KNN_model import *
# from data_loader0_1 import *
from data_loader0_1_revision import *
from earlystop import EarlyStopping


def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    print(confusion)
    ppv = precision_score(y_test, pred)
    npv = precision_score(1-y_test, 1-pred)
    sensitivity = recall_score(y_test, pred)
    return ppv, npv, sensitivity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="/home/daehyun/KNN_final/only12/final_knn_dataset.csv")
    parser.add_argument('--outputs-dir', type=str, default="/home/daehyun/KNN_final/only12/weight_revision")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir)
    out_path = args.outputs_dir

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # torch.manual_seed(123)

    model = KNN5().to(device)
    learning_rate = args.lr

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_dataset = TrainDataset_necip(args.data)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=False,
                                  drop_last=True)

    eval_dataset1 = EvalDataset1_necip(args.data)
    eval_dataloader1 = DataLoader(dataset=eval_dataset1, batch_size=1)
    eval_dataset2 = EvalDataset2_necip(args.data)
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

        fpr, tpr, thresholds = roc_curve(label, predict1)
        J = tpr - fpr
        # J = sqrt(tpr * (1-fpr))
        ix = argmax(J)

        best_threshold = thresholds[ix]
        print(best_threshold)
        predict1 = np.array(predict1)
        label = np.array(label)

        binarizer = Binarizer(threshold=best_threshold)
        custom_predict = binarizer.fit_transform(predict1.reshape(-1, 1))

        ppv, npv, sensitivity = get_clf_eval(label, custom_predict)

        print("EPOCH : {0:3d}  ntety_loss : {1:0.4f}  AUC_ntety : {2:0.4f}, ppv : {3:0.4f}, npv : {4:0.4f}, sensitivity : {5:0.4f}"
        .format(epoch, testloss, auc_acc, ppv, npv, sensitivity))

        if epoch > 5:
            early_stopping(testloss, model)
            
            if min_loss > testloss:
                min_loss = testloss

        if early_stopping.early_stop:
            print('stop!')
            print('minimun loss is ', min_loss)
            break

