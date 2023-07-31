#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：LPIGLAM 
@File    ：ShowResult.py
@IDE     ：PyCharm 
@Author  ：Shao shaoqi
@Date    ：28/7/2023 下午3:36 
'''
import numpy as np


def show_result(DATASET, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List, Sn_List, Sp_List, Mcc_List, Ensemble=False):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(
        Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    Sn_mean, Sn_var = np.mean(Sn_List), np.var(Sn_List)
    Sp_mean, Sp_var = np.mean(Sp_List), np.var(Sp_List)
    Mcc_mean, Mcc_var = np.mean(Mcc_List), np.var(Mcc_List)

    if Ensemble == False:
        print("The model's results:")
        filepath = "./{}/results.txt".format(DATASET)
    else:
        print("The ensemble model's results:")
        filepath = "./{}/ensemble_results.txt".format(DATASET)
    with open(filepath, 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(
            Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(
            Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(
            Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')
    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(
        Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))
    print('Sn(std):{:.4f}({:.4f})'.format(Sn_mean, Sn_var))
    print('Sp(std):{:.4f}({:.4f})'.format(Sp_mean, Sp_var))
    print('Mcc(std):{:.4f}({:.4f})'.format(Mcc_mean, Mcc_var))
