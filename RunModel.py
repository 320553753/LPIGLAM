#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：LPIGLAM 
@File    ：RunModel.py
@IDE     ：PyCharm 
@Author  ：Shao shaoqi
@Date    ：28/7/2023 下午2:37 
'''
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from prefetch_generator import BackgroundGenerator

from torch.utils.data import DataLoader
from tqdm import tqdm

from Config import hyperparameter
from Model import LPIGLAM
from Utils.DataPrepare import get_kfold_data, shuffle_dataset
from Utils.DataSetsFunction import CustomDataSet, collate_fn
from LossFunction import CELoss
from Utils.TestModel import test_model
from Utils.ShowResult import show_result

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_model(SEED, DATASET, MODEL, K_Fold, LOSS):
    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load dataset from text file'''
    assert DATASET in ["ATH","ZEA","RPI369","RPI488","RPI1807","RPI2241","NPInter"]
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")
    '''set loss function weight'''

    weight_loss = None

    '''shuffle data'''
    print("data shuffle")
    data_list = shuffle_dataset(data_list, SEED)

    '''split dataset to train&validation set and test set'''
    train_data_list = data_list
    print('Number of Train&Val set: {}'.format(len(train_data_list)))

    '''metrics'''
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []
    Sn_List_stable,Sp_List_stable,Mcc_List_stable=[],[],[]
    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        train_dataset, valid_dataset = get_kfold_data(i_fold, train_data_list, k=K_Fold)

        train_dataset = CustomDataSet(train_dataset)
        valid_dataset = CustomDataSet(valid_dataset)
        train_size = len(train_dataset)

        train_dataset_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                          collate_fn=collate_fn, drop_last=True)
        valid_dataset_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                          collate_fn=collate_fn, drop_last=True)


        """ create model"""
        model = MODEL(hp).to(DEVICE)

        print("Initialize weights")
        """Initialize weights"""


        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]




        """create optimizer and scheduler"""
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // hp.Batch_size)

        Loss = CELoss(weight_CE=weight_loss, DEVICE=DEVICE)

        """Output files"""
        save_path = "./" + DATASET + "/{}".format(i_fold+1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'

        """Start training."""
        print("start training")

        for epoch in range(1, hp.Epoch + 1):


            print("====================epoch:",epoch,"===================")
            train_pbar = tqdm(
                enumerate(BackgroundGenerator(train_dataset_loader)),
                total=len(train_dataset_loader))

            """train"""
            train_losses_in_epoch = []
            model.train()
            for train_i, train_data in train_pbar:
                train_lncRNA,train_lncRNA_kmer, train_proteins,train_proteins_kmer, train_labels = train_data
                train_lncRNA = train_lncRNA.to(DEVICE)
                train_lncRNA_kmer=train_lncRNA_kmer.to(DEVICE)
                train_proteins = train_proteins.to(DEVICE)
                train_proteins_kmer=train_proteins_kmer.to(DEVICE)
                train_labels = train_labels.to(DEVICE)

                optimizer.zero_grad()

                predicted_interaction = model(train_lncRNA, train_proteins,train_lncRNA_kmer,train_proteins_kmer)
                train_loss = Loss(predicted_interaction, train_labels)
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(
                train_losses_in_epoch)  # 一次epoch的平均训练loss

            print("loss:",train_loss_a_epoch)




        '''test model'''
        trainset_test_stable_results, _, _, _, _, _, _, _, _ = test_model(
            model, train_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Train", FOLD_NUM=1)
        testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test, Sn_test, Sp_test, Mcc_test = test_model(
            model, valid_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Valid", FOLD_NUM=1)
        AUC_List_stable.append(AUC_test)
        Accuracy_List_stable.append(Accuracy_test)
        AUPR_List_stable.append(PRC_test)
        Recall_List_stable.append(Recall_test)
        Precision_List_stable.append(Precision_test)
        Sn_List_stable.append(Sn_test)
        Sp_List_stable.append(Sp_test)
        Mcc_List_stable.append(Mcc_test)
        with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')

    show_result(DATASET, Accuracy_List_stable, Precision_List_stable,
                Recall_List_stable, AUC_List_stable, AUPR_List_stable,Sn_List_stable,Sp_List_stable,Mcc_List_stable, Ensemble=False)

