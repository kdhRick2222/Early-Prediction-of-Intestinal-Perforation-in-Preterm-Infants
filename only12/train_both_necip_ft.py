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

from KNN_model import use_ntet, sharing_net
from data_loader_both import *
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
    parser.add_argument('--lr2', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--limit', type=float, default=0.878)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir)
    out_path = args.outputs_dir
    limit = args.limit

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(123)

    #model = sharing_net().to(device)
    model = use_ntet().to(device)
    learning_rate = args.lr
    lr2 = args.lr2

    criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    early_stopping = EarlyStopping(patience=5, verbose=True)
    early_stopping2 = EarlyStopping(patience=7, verbose=True)

    train_dataset_ntet = TrainDataset_nec(args.data)
    train_dataloader_ntet = DataLoader(dataset=train_dataset_ntet,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=False,
                                  drop_last=True)

    eval_dataset1_ntet = EvalDataset1_nec(args.data)
    eval_dataloader1_ntet = DataLoader(dataset=eval_dataset1_ntet, batch_size=1)
    eval_dataset2_ntet = EvalDataset2_nec(args.data)
    eval_dataloader2_ntet = DataLoader(dataset=eval_dataset2_ntet, batch_size=1)

    train_dataset = TrainDataset_necip(args.data)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=False,
                                  drop_last=True)

    eval_dataset1 = EvalDataset1_necip(args.data)
    eval_dataloader1 = DataLoader(dataset=eval_dataset1, batch_size=1)
    eval_dataset2 = EvalDataset2_necip(args.data)
    eval_dataloader2 = DataLoader(dataset=eval_dataset2, batch_size=1)

    
    min_loss = 10.

    for epoch in range(args.num_epochs):

        model.train()
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        loss_sum1 = 0
        loss_sum2 = 0

        for data in train_dataloader_ntet:
            
            inputs, labels1_, labels2_ = data
            
            inputs = inputs.to(device)
            labels1_ = labels1_.to(device)
            labels2_ = labels2_.to(device)

            labels1 = labels1_.unsqueeze(1)
            labels2 = labels2_.unsqueeze(1)


            #input_ = torch.cat((inputs[:, :50], inputs[:, 53:54]), dim=1)
            #print(input_)
            
            ntet, ntety = model(inputs)

            loss1 = criterion(ntet, labels1)
            loss2 = criterion(ntety, labels2)
            loss = loss1 + loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum1 += loss1
            loss_sum2 += loss2
 
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'train_both_{}.pth'.format(epoch)))

        model.eval()

        ntet1_acc, ntet2_acc, ntety1_acc, ntety2_acc = 0, 0, 0, 0
        ntet_loss1, ntet_loss2, ntety_loss1, ntety_loss2 = 0, 0, 0, 0

        n11, n12, n21, n22 = 0, 0, 0, 0

        label = []
        label_ = []
        predict1 = []
        predict2 = []

        for data in eval_dataloader1_ntet:

            inputs, labels1_, labels2_ = data

            inputs = inputs.to(device)
            labels1_ = labels1_.to(device)
            labels2_ = labels2_.to(device)

            #input_ = torch.cat((inputs[:, :50], inputs[:, 53:54]), dim=1)

            with torch.no_grad():
                ntet, ntety = model(inputs)
            
            label += labels1_.tolist()
            label_ += labels2_.tolist()
            predict1 += ntet.tolist()
            predict2 += ntety.tolist()

            labels1 = labels1_ #.unsqueeze(1)
            labels2 = labels2_ #.unsqueeze(1)
            
            ntet_loss1 += criterion(ntet, labels1)
            ntety_loss1 += criterion(ntety, labels2)

            if ntet > 0.5:
                ntet = 1
            else:
                ntet = 0
            
            if labels1 == 0:
                n11 += 1
                if ntet == 0:
                    ntet1_acc += 1
                else:
                    ntet1_acc += 0
            else:
                n12 += 1
                if ntet == 1:
                    ntet2_acc += 1
                else:
                    ntet2_acc += 0
            
            if ntety > 0.5:
                ntety = 1
            else:
                ntety = 0
            
            if ntety == 0:
                ntety1_acc += 1
            else:
                ntety1_acc += 0

        for data in eval_dataloader2_ntet:

            inputs, labels1_, labels2_ = data

            inputs = inputs.to(device)
            labels1_ = labels1_.to(device)
            labels2_ = labels2_.to(device)

            #input_ = torch.cat((inputs[:, :50], inputs[:, 53:54]), dim=1)

            with torch.no_grad():
                ntet, ntety = model(inputs)

            label += labels1_.tolist()
            label_ += labels2_.tolist()
            predict1 += ntet.tolist()
            predict2 += ntety.tolist()

            labels1 = labels1_ #.unsqueeze(1)
            labels2 = labels2_ #.unsqueeze(1)
            
            ntet_loss2 += criterion(ntet, labels1)
            ntety_loss2 += criterion(ntety, labels2)

            if ntet > 0.5:
                ntet = 1
            else:
                ntet = 0

            if labels1 == 0:
                n21 += 1
                if ntet == 0:
                    ntet1_acc += 1
                else:
                    ntet1_acc += 0
            else:
                n22 += 1
                if ntet == 1:
                    ntet2_acc += 1
                else:
                    ntet2_acc += 0
            
            if ntety > 0.5:
                ntety = 1
            else:
                ntety = 0
            
            if ntety == 1:
                ntety2_acc += 1
            else:
                ntety2_acc += 0
            

        n1 = n11 + n12
        n2 = n21 + n22
        
        ntet_loss = (ntet_loss1 + ntet_loss2) / (n1 + n2)
        ntety_loss = ntety_loss1/n1 + ntety_loss2/n2
        test_loss = ntet_loss + ntety_loss

        label = np.array(label)        
        label_ = np.array(label_)
        predict1 = np.array(predict1)
        predict2 = np.array(predict2)

        auc_ntet = roc_auc_score(label, predict1)
        auc_ntety = roc_auc_score(label_, predict2)

        fpr, tpr, thresholds = roc_curve(label_, predict2)
        J = tpr - fpr
        # J = sqrt(tpr * (1-fpr))
        ix = argmax(J)

        best_threshold = thresholds[ix]
        print(best_threshold)

        binarizer = Binarizer(threshold=best_threshold)
        custom_predict = binarizer.fit_transform(predict2.reshape(-1, 1))

        ppv, npv, sensitivity = get_clf_eval(label_, custom_predict)

        if epoch == 0:
            print(n11, n12, n1, n21, n22, n2)

        print("EPOCH : {0:3d}  ntety_loss : {1:0.4f}  AUC_ntet : {2:0.4f}  AUC_ntety : {3:0.4f}, ppv(ntety) : {4:0.4f}, npv(ntety) : {5:0.4f}, sensitivity : {6:0.4f}"
        .format(epoch, ntety_loss, auc_ntet, auc_ntety, ppv, npv, sensitivity))

        if limit < auc_ntety:
            break


    for epoch in range(args.num_epochs):

        model.train()
        optimizer = optim.SGD(model.parameters(), lr=lr2, momentum=0.9)

        loss_sum1 = 0
        loss_sum2 = 0

        for data in train_dataloader:
            
            inputs, labels1_, labels2_ = data
            
            inputs = inputs.to(device)
            labels1_ = labels1_.to(device)
            labels2_ = labels2_.to(device)

            labels1 = labels1_.unsqueeze(1)
            labels2 = labels2_.unsqueeze(1)
            
            ntet, ntety = model(inputs)

            loss1 = criterion(ntet, labels1)
            loss2 = criterion(ntety, labels2)
            loss = loss2+loss1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum1 += loss1
            loss_sum2 += loss2
 
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'train_both_{}.pth'.format(epoch)))

        model.eval()

        ntet1_acc, ntet2_acc, ntety1_acc, ntety2_acc = 0, 0, 0, 0
        ntet_loss1, ntet_loss2, ntety_loss1, ntety_loss2 = 0, 0, 0, 0

        n11, n12, n21, n22 = 0, 0, 0, 0

        label = []
        label_ = []
        predict1 = []
        predict2 = []

        for data in eval_dataloader1:

            inputs, labels1_, labels2_ = data

            inputs = inputs.to(device)
            labels1_ = labels1_.to(device)
            labels2_ = labels2_.to(device)

            with torch.no_grad():
                ntet, ntety = model(inputs)
            
            label += labels1_.tolist()
            label_ += labels2_.tolist()
            predict1 += ntet.tolist()
            predict2 += ntety.tolist()

            labels1 = labels1_ #.unsqueeze(1)
            labels2 = labels2_ #.unsqueeze(1)
            
            ntet_loss1 += criterion(ntet, labels1)
            ntety_loss1 += criterion(ntety, labels2)

            if ntet > 0.5:
                ntet = 1
            else:
                ntet = 0
            
            if labels1 == 0:
                n11 += 1
                if ntet == 0:
                    ntet1_acc += 1
                else:
                    ntet1_acc += 0
            else:
                n12 += 1
                if ntet == 1:
                    ntet2_acc += 1
                else:
                    ntet2_acc += 0
            
            if ntety > 0.5:
                ntety = 1
            else:
                ntety = 0
            
            if ntety == 0:
                ntety1_acc += 1
            else:
                ntety1_acc += 0

        for data in eval_dataloader2:

            inputs, labels1_, labels2_ = data

            inputs = inputs.to(device)
            labels1_ = labels1_.to(device)
            labels2_ = labels2_.to(device)

            with torch.no_grad():
                ntet, ntety = model(inputs)

            label += labels1_.tolist()
            label_ += labels2_.tolist()
            predict1 += ntet.tolist()
            predict2 += ntety.tolist()

            labels1 = labels1_ #.unsqueeze(1)
            labels2 = labels2_ #.unsqueeze(1)
            
            ntet_loss2 += criterion(ntet, labels1)
            ntety_loss2 += criterion(ntety, labels2)

            if ntet > 0.5:
                ntet = 1
            else:
                ntet = 0

            if labels1 == 0:
                n21 += 1
                if ntet == 0:
                    ntet1_acc += 1
                else:
                    ntet1_acc += 0
            else:
                n22 += 1
                if ntet == 1:
                    ntet2_acc += 1
                else:
                    ntet2_acc += 0
            
            if ntety > 0.5:
                ntety = 1
            else:
                ntety = 0
            
            if ntety == 1:
                ntety2_acc += 1
            else:
                ntety2_acc += 0
            

        n1 = n11 + n12
        n2 = n21 + n22
        
        ntet_loss = (ntet_loss1 + ntet_loss2) / (n1 + n2)
        ntety_loss = ntety_loss1/n1 + ntety_loss2/n2
        test_loss = ntety_loss# ntet_loss + ntety_loss

        label = np.array(label)        
        label_ = np.array(label_)
        predict1 = np.array(predict1)
        predict2 = np.array(predict2)

        auc_ntet = roc_auc_score(label, predict1)
        auc_ntety = roc_auc_score(label_, predict2)

        fpr, tpr, thresholds = roc_curve(label_, predict2)
        J = tpr - fpr
        ix = argmax(J)

        best_threshold = thresholds[ix]
        print(best_threshold)

        binarizer = Binarizer(threshold=best_threshold)
        custom_predict = binarizer.fit_transform(predict2.reshape(-1, 1))

        ppv, npv, sensitivity = get_clf_eval(label_, custom_predict)

        if epoch == 0:
            print(n11, n12, n1, n21, n22, n2)

        print("EPOCH : {0:3d}  ntety_loss : {1:0.4f}  AUC_ntet : {2:0.4f}  AUC_ntety : {3:0.4f}, ppv(ntety) : {4:0.4f}, npv(ntety) : {5:0.4f}, sensitivity : {6:0.4f}"
        .format(epoch, ntety_loss, auc_ntet, auc_ntety, ppv, npv, sensitivity))
        
        if epoch > 0:
            early_stopping2(test_loss, model)
            
            if min_loss > test_loss:
                min_loss = test_loss

        if early_stopping2.early_stop:
            print('stop!')
            print('minimun loss is ', min_loss)
            break

