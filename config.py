#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：LPIGLAM 
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：Shao shaoqi
@Date    ：28/7/2023 下午2:38 
'''

class hyperparameter():
    def __init__(self):
        self.Learning_rate = 1e-4
        self.Epoch = 200
        self.Batch_size = 4
        self.Patience = 50
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.embed_dim = 64
        self.protein_kernel = [5, 9, 13]
        self.lncRNA_kernel = [5, 7, 9]
        self.conv = 40
        self.char_dim = 64
        self.loss_epsilon = 1