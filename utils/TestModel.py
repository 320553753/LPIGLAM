#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：LPIGLAM 
@File    ：TestModel.py
@IDE     ：PyCharm 
@Author  ：Shao shaoqi
@Date    ：28/7/2023 下午3:36 
'''
import math
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve)

# calculate the 3 metrics of  Sn, Sp, MCC
def calc_metrics(y_label, y_proba):
    con_matrix = confusion_matrix(y_label, y_proba)
    TN = float(con_matrix[0][0])
    FP = float(con_matrix[0][1])
    FN = float(con_matrix[1][0])
    TP = float(con_matrix[1][1])
    P = TP + FN
    N = TN + FP
    Sn = TP / P
    Sp = TN / N

    MCC = 0
    tmp = math.sqrt((TP + FP) * (TP + FN)) * math.sqrt((TN + FP) * (TN + FN))
    if tmp != 0:
        MCC = (TP * TN - FP * FN) / tmp

    return  Sn, Sp, MCC


def test_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM):
    if isinstance(MODEL, list):
        for item in MODEL:
            item.eval()
    else:
        MODEL.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            lncRNA,lncRNA_kmer, proteins,proteins_kmer, labels = data
            lncRNA = lncRNA.to(DEVICE)
            lncRNA_kmer=lncRNA_kmer.to(DEVICE)
            proteins = proteins.to(DEVICE)
            proteins_kmer=proteins_kmer.to(DEVICE)
            labels = labels.to(DEVICE)

            if isinstance(MODEL, list):
                predicted_scores = torch.zeros(2).to(DEVICE)
                for i in range(len(MODEL)):
                    predicted_scores = predicted_scores + \
                        MODEL[i](lncRNA, proteins, lncRNA_kmer,proteins_kmer)
                predicted_scores = predicted_scores / FOLD_NUM
            else:
                predicted_scores = MODEL(lncRNA, proteins,lncRNA_kmer,proteins_kmer)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(
                predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Recall = recall_score(Y, P)
    #AUC = roc_auc_score(Y, S)
    fpr_CSNN, tpr_CSNN, thresholds = roc_curve(Y,S)
    AUC = auc(fpr_CSNN, tpr_CSNN)

    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)
    Sn,Sp,Mcc=calc_metrics(Y,P)

    plt.rc('font', family='Times New Roman')

    plt.plot(fpr_CSNN, tpr_CSNN, 'purple', label='NN_AUC = %0.2f' % AUC)
    plt.legend(loc='lower right', fontsize=12)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xlabel('Flase Positive Rate', fontsize=14)
    plt.show()

    return Y, P, test_loss, Accuracy, Precision, Recall, AUC, PRC,Sn,Sp,Mcc


def test_model(MODEL, dataset_loader, save_path, DATASET, LOSS, DEVICE, dataset_class="Train", save=True, FOLD_NUM=1):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_loader)),
        total=len(dataset_loader))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test, Sn_test, Sp_test, Mcc_test = test_precess(
        MODEL, test_pbar, LOSS, DEVICE, FOLD_NUM)
    if save:
        if FOLD_NUM == 1:
            filepath = save_path + \
                "/{}_{}_prediction.txt".format(DATASET, dataset_class)
        else:
            filepath = save_path + \
                "/{}_{}_ensemble_prediction.txt".format(DATASET, dataset_class)
        with open(filepath, 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f};Sn:{:.5f};Sp:{:.5f};Mcc:{:.5f}.' \
        .format(dataset_class, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test,Sn_test,Sp_test,Mcc_test)
    print(results)
    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test,Sn_test,Sp_test,Mcc_test
