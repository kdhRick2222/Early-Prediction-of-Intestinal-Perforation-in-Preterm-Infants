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

from KNN_model import KNN
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
#sip 0.8019

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_file', type=str, default="/home/daehyun/KNN_final/only12/weight_re/nec_alone.pth")
    # parser.add_argument('--weights_file', type=str, default="/home/daehyun/KNN_final/only12/weight/nec_for_transfer.pth")
    parser.add_argument('--data', type=str, default="/home/daehyun/KNN_final/only12/final_knn_dataset.csv")
    parser.add_argument('--outputs-dir', type=str, default="/home/daehyun/KNN_final/only12/weight_revision")
    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--batch-size', type=int, default=2048)#1e-5,64, 8004 // 1e-4, 2048, 8022
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=64)#1e-5,64, 8004 // 1e-4, 2048, 8022
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir)
    out_path = args.outputs_dir

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = KNN().to(device)
    model.load_state_dict(torch.load(args.weights_file))

    # for para in model.parameters():
    #     para.requires_grad = False
    
    # for name, param in model.named_parameters():
    #     if name in ['linear6.weight', 'linear6.bias', 'linear5.weight', 'linear5.bias', 'linear4.weight', 'linear4.bias']:
    #         param.requires_grad = True

    learning_rate = args.lr

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train_dataset1 = TrainDataset_sip(args.data)
    train_dataloader = DataLoader(dataset=train_dataset1,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset1 = EvalDataset1_sip(args.data)
    eval_dataloader1 = DataLoader(dataset=eval_dataset1, batch_size=1)
    eval_dataset2 = EvalDataset2_sip(args.data)
    eval_dataloader2 = DataLoader(dataset=eval_dataset2, batch_size=1)


    for epoch in range(args.num_epochs):

        model.train()

        loss = 0

        for data in train_dataloader:
            
            inputs, labels = data
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            # labels = labels.to(device).reshape(-1,1).float()
            
            preds = model(inputs)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'transfer_{}.pt'.format(epoch)))
        
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
            # labels = labels.to(device).reshape(-1,1).float()

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
        
        testloss = testloss1/n1 + testloss2/n2
        a1 = accuracy1 / n1
        a2 = accuracy2 / n2
        acc = (a1 + a2) * 0.5
        auc_acc = roc_auc_score(label, predict1)

        fpr, tpr, thresholds = roc_curve(label, predict1)
        J = tpr - fpr
        # J = sqrt(tpr * (1-fpr))
        ix = argmax(J)

        # best_threshold = thresholds[ix]
        best_threshold = 0.5
        print(best_threshold)
        predict1 = np.array(predict1)
        label = np.array(label)

        binarizer = Binarizer(threshold=best_threshold)
        custom_predict = binarizer.fit_transform(predict1.reshape(-1, 1))

        ppv, npv, sensitivity = get_clf_eval(label, custom_predict)

        print("EPOCH : {0:3d}  Loss : {1:0.4f}  AUC : {2:0.4f}, ppv : {3:0.4f}, npv : {4:0.4f}, sensitivity : {5:0.4f}"
        .format(epoch, testloss, auc_acc, ppv, npv, sensitivity))