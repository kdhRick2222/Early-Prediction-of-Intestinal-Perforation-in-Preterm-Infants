import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import sklearn
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Binarizer
from numpy import argmax

from models import *
from data_loader0_1 import *
from data_loader_both import *


def roc_curve_plot(l1, l2, p1, p2, p3):
    fprs1, tprs1, thresholds1 = roc_curve(l1, p1)
    fprs2, tprs2, thresholds2 = roc_curve(l2, p2)
    fprs3, tprs3, thresholds3 = roc_curve(l1, p3)

    plt.clf()

    plt.plot(fprs1, tprs1, label='Model1')
    # plt.plot(fprs2, tprs2, label='Model2')
    # plt.plot(fprs3, tprs3, label='Model3')
    plt.plot([0,1], [0,1], 'k--', label='Random')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.xlabel('FPR( 1 - Specificity  )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.savefig('/home/daehyun/KNN_final/roc_curve_12/roc/nec_roc_curve.png')


def get_clf_eval(y_test, pred):
    cm = confusion_matrix(y_test, pred)
    # print(cm)
    ppv = precision_score(y_test, pred)
    npv = precision_score(1-y_test, 1-pred)
    specificity = recall_score(y_test, pred, pos_label=0)
    sensitivity = recall_score(y_test, pred)
    return ppv, npv, sensitivity, specificity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1_weight', type=str, default="/home/daehyun/KNN_final/only12/weight_re/nec_alone.pth")
    parser.add_argument('--model2_weight', type=str, default="/home/daehyun/KNN/roc_curve_12/weight_12/nec_both.pth")
    parser.add_argument('--model3_weight', type=str, default="/home/daehyun/KNN/roc_curve_12/weight_12/nec_transfer.pth")
    parser.add_argument('--data', type=str, default="/home/daehyun/KNN_final/only12/final_knn_dataset.csv")
    # parser.add_argument('--data', type=str, default="/home/daehyun/KNN/KNN_test_data.csv")
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model1 = KNN().to(device)
    model1.load_state_dict(torch.load(args.model1_weight))
    model2 = use_ntet().to(device)
    model2.load_state_dict(torch.load(args.model2_weight))
    model3 = KNN().to(device)
    model3.load_state_dict(torch.load(args.model3_weight))

    criterion = nn.BCELoss()

    eval_dataset1 = EvalDataset1_nec(args.data)
    eval_dataloader1 = DataLoader(dataset=eval_dataset1, batch_size=1)
    eval_dataset2 = EvalDataset2_nec(args.data)
    eval_dataloader2 = DataLoader(dataset=eval_dataset2, batch_size=1)

    eval_dataset1_both = EvalDataset1_nec_both(args.data)
    eval_dataloader1_both = DataLoader(dataset=eval_dataset1_both, batch_size=1)
    eval_dataset2_both = EvalDataset2_nec_both(args.data)
    eval_dataloader2_both = DataLoader(dataset=eval_dataset2_both, batch_size=1)

    label1, label2, predict1, predict2, predict3 = [], [], [], [], []

    model1.eval()
    model2.eval()
    model3.eval()

    for data in eval_dataloader1:

        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds1 = model1(inputs)
            preds3 = model3(inputs)

        label1 += labels.tolist()
        predict1 += preds1.tolist()
        predict3 += preds3.tolist()

    for data in eval_dataloader2:

        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds1 = model1(inputs)
            preds3 = model3(inputs)

        label1 += labels.tolist()
        predict1 += preds1.tolist()
        predict3 += preds3.tolist()

    
    for data in eval_dataloader1_both:

        inputs, labels1_, labels2_ = data

        inputs = inputs.to(device)
        labels1_ = labels1_.to(device)
        labels2_ = labels2_.to(device)

        with torch.no_grad():
            ntety, ntet = model2(inputs)
        
        label2 += labels2_.tolist()
        predict2 += ntet.tolist()


    for data in eval_dataloader2_both:

        inputs, labels1_, labels2_ = data

        inputs = inputs.to(device)
        labels1_ = labels1_.to(device)
        labels2_ = labels2_.to(device)

        with torch.no_grad():
            ntety, ntet = model2(inputs)

        label2 += labels1_.tolist()
        predict2 += ntet.tolist()

    # p1 = np.array(predict1)
    # predict1_ = np.where(p1>0.6659, 1, 0)
    # p2 = np.array(predict2)
    # predict2_ = np.where(p2>0.6048, 1, 0)
    # p3 = np.array(predict3)
    # predict3_ = np.where(p3>0.3437, 1, 0)
    auc_1 = roc_auc_score(label1, predict1)
    auc_2 = roc_auc_score(label2, predict2)
    auc_3 = roc_auc_score(label1, predict3)

    roc_curve_plot(label1, label2, predict1, predict2, predict3)
    
    values = []
    
    fpr1, tpr1, thresholds1 = roc_curve(label1, predict1)
    fpr2, tpr2, thresholds2 = roc_curve(label2, predict2)
    fpr3, tpr3, thresholds3 = roc_curve(label1, predict3)

    J1 = tpr1 - fpr1
    ix1 = argmax(J1)
    best_threshold1 = thresholds1[ix1]
    predict1 = np.array(predict1)
    label1 = np.array(label1)
    binarizer1 = Binarizer(threshold=best_threshold1)
    custom_predict1 = binarizer1.fit_transform(predict1.reshape(-1, 1))
    ppv1, npv1, sensitivity1, specificity1 = get_clf_eval(label1, custom_predict1)
    f1_score_1 = f1_score(label1, custom_predict1, labels=None, average='weighted')
    values.append([auc_1, f1_score_1, ppv1, npv1, sensitivity1, specificity1])
    # values.append([auc_1, f1_score_1])

    J2 = tpr2 - fpr2
    ix2 = argmax(J2)
    best_threshold2 = thresholds2[ix2]
    predict2 = np.array(predict2)
    label2 = np.array(label2)
    binarizer2 = Binarizer(threshold=best_threshold2)
    custom_predict2 = binarizer2.fit_transform(predict2.reshape(-1, 1))
    ppv2, npv2, sensitivity2,  specificity2 = get_clf_eval(label2, custom_predict2)
    f1_score_2 = f1_score(label2, custom_predict2, labels=None, average='macro')
    values.append([auc_2, f1_score_2, ppv2, npv2, sensitivity2, specificity2])
    # values.append([auc_2, f1_score_2])

    J3 = tpr3 - fpr3
    ix3 = argmax(J3)
    best_threshold3 = thresholds3[ix3]
    predict3 = np.array(predict3)
    label3 = np.array(label1)
    binarizer3 = Binarizer(threshold=best_threshold3)
    custom_predict3 = binarizer3.fit_transform(predict3.reshape(-1, 1))
    ppv3, npv3, sensitivity3, specificity3 = get_clf_eval(label3, custom_predict3)
    f1_score_3 = f1_score(label3, custom_predict3, labels=None, average='macro')
    values.append([auc_3, f1_score_3, ppv3, npv3, sensitivity3, specificity3])
    # values.append([auc_3, f1_score_3])

    models_dataframe=pd.DataFrame(values,index=['model1', 'model2', 'model3'])
    models_dataframe.columns=['AUC', 'f1_score', 'ppv', 'npv', 'sensitivity', 'specificity']
    # models_dataframe.columns=['AUC', 'f1_score']

    print('Threshold1 : ', best_threshold1)
    print('Threshold2 : ', best_threshold2)
    print('Threshold3 : ', best_threshold3)

    print(models_dataframe)