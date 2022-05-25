import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Binarizer
from numpy import argmax

from models import *
from test_loader0_1 import *
from test_loader_both import *


def roc_curve_plot(l1, l2, p1, p2, p3):
    fprs1, tprs1, thresholds1 = roc_curve(l1, p1)
    fprs2, tprs2, thresholds2 = roc_curve(l2, p2)
    fprs3, tprs3, thresholds3 = roc_curve(l1, p3)

    plt.clf()

    plt.plot(fprs1, tprs1, label='Model1')
    plt.plot(fprs2, tprs2, label='Model2')
    plt.plot(fprs3, tprs3, label='Model3')
    plt.plot([0,1], [0,1], 'k--', label='Random')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.savefig('/home/daehyun/KNN/test_results/roc/ip_roc_curve.png')


def get_clf_eval(y_test, pred):
    # cm = confusion_matrix(y_test, pred)
    ppv = precision_score(y_test, pred)
    npv = precision_score(1-y_test, 1-pred)
    sensitivity = recall_score(y_test, pred)
    return ppv, npv, sensitivity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model1_weight', type=str, default="/home/daehyun/KNN/roc_curve_12/weight_12/ip_alone.pth")
    # parser.add_argument('--model2_weight', type=str, default="/home/daehyun/KNN/roc_curve_12/weight_12/ip_both.pth")
    # parser.add_argument('--model3_weight', type=str, default="/home/daehyun/KNN/roc_curve_12/weight_12/ip_transfer.pth")
    parser.add_argument('--model1_weight', type=str, default="/home/daehyun/KNN/roc_curve_12/weight_re/ip_alone.pth")
    parser.add_argument('--model2_weight', type=str, default="/home/daehyun/KNN/roc_curve_12/weight_re/ip_both.pth")
    parser.add_argument('--model3_weight', type=str, default="/home/daehyun/KNN/roc_curve_12/weight_re/ip_transfer.pth")
    parser.add_argument('--data', type=str, default="/home/daehyun/KNN/only12/KNNdata_only12.csv")
    # parser.add_argument('--test', type=str, default="/home/daehyun/KNN/test_results/test_dataset2019_60.csv") #60개 데이터
    parser.add_argument('--test', type=str, default="/home/daehyun/KNN/test_results/test_dataset2019_57.csv") #57개 데이터
    # parser.add_argument('--test', type=str, default="/home/daehyun/KNN/KNN_test.csv") #26개 데이터
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

    eval_dataset = EvalDataset_ip(args.data, args.test)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    eval_dataset_both = EvalDataset_ip_both(args.data, args.test)
    eval_dataloader_both = DataLoader(dataset=eval_dataset_both, batch_size=1)


    label1, label2, predict1, predict2, predict3 = [], [], [], [], []
    answer = []

    model1.eval()
    model2.eval()
    model3.eval()

    for data in eval_dataloader:

        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds1 = model1(inputs)
            preds3 = model3(inputs)

        label1 += labels.tolist()
        predict1 += preds1.tolist()
        predict3 += preds3.tolist()
        answer += labels.tolist()
    
    for data in eval_dataloader_both:

        inputs, labels1_, labels2_ = data

        inputs = inputs.to(device)
        labels1_ = labels1_.to(device)
        labels2_ = labels2_.to(device)

        with torch.no_grad():
            ntety, ntet = model2(inputs)
        
        label2 += labels2_.tolist()
        predict2 += ntet.tolist()

    # l1 = np.array(label1)
    # l2 = np.array(label2)
    # label1 = 1-l1
    # label2 = 1-l2
   
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
    # best_threshold1 = thresholds1[ix1]
    best_threshold1 = 0.3087265193462372
    predict1 = np.array(predict1)
    label1 = np.array(label1)
    binarizer1 = Binarizer(threshold=best_threshold1)
    custom_predict1 = binarizer1.fit_transform(predict1.reshape(-1, 1))
    auc_1 = roc_auc_score(label1, custom_predict1)
    ppv1, npv1, sensitivity1 = get_clf_eval(label1, custom_predict1)
    values.append([auc_1, ppv1, npv1, sensitivity1])

    J2 = tpr2 - fpr2
    ix2 = argmax(J2)
    # best_threshold2 = thresholds2[ix2]
    best_threshold2 = 0.3363470733165741
    predict2 = np.array(predict2)
    label2 = np.array(label2)
    binarizer2 = Binarizer(threshold=best_threshold2)
    custom_predict2 = binarizer2.fit_transform(predict2.reshape(-1, 1))
    auc_2 = roc_auc_score(label2, custom_predict2)
    ppv2, npv2, sensitivity2 = get_clf_eval(label2, custom_predict2)
    values.append([auc_2, ppv2, npv2, sensitivity2])

    J3 = tpr3 - fpr3
    ix3 = argmax(J3)
    # best_threshold3 = thresholds3[ix3]
    best_threshold3 = 0.4692429304122925
    predict3 = np.array(predict3)
    label3 = np.array(label1)
    binarizer3 = Binarizer(threshold=best_threshold3)
    custom_predict3 = binarizer3.fit_transform(predict3.reshape(-1, 1))
    auc_3 = roc_auc_score(label1, custom_predict3)
    ppv3, npv3, sensitivity3 = get_clf_eval(label3, custom_predict3)
    values.append([auc_3, ppv3, npv3, sensitivity3])

    models_dataframe=pd.DataFrame(values,index=['model1', 'model2', 'model3'])
    models_dataframe.columns=['AUC', 'ppv', 'npv', 'sensitivity']

    print(models_dataframe)
    
    p1 = np.array(predict1).squeeze()
    p2 = np.array(predict2).squeeze()
    p3 = np.array(predict3).squeeze()
    answer_ = np.array(answer).squeeze()
    probability1, probability2, probability3 = [], [], []

    for i in range(len(p1)):
        probability1.append(round(p1[i], 2))
        probability2.append(round(p2[i], 2))
        probability3.append(round(p3[i], 2))

    # print("Answer: ", answer_)
    # print("Model1: ", probability1)
    # print("Model2: ", probability2)
    # print("Model3: ", probability3)