#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：LPIGLAM 
@File    ：DataPrepare.py
@IDE     ：PyCharm 
@Author  ：Shao shaoqi
@Date    ：28/7/2023 下午3:35 
'''

import numpy as np


def get_kfold_data(i, datasets, k=5):
    fold_size = len(datasets) // k

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:]
        trainset = datasets[0:val_start]

    return trainset, validset


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
