#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：LPIGLAM 
@File    ：LossFunction.py
@IDE     ：PyCharm 
@Author  ：Shao shaoqi
@Date    ：28/7/2023 下午2:39 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F



class CELoss(nn.Module):
    def __init__(self, weight_CE, DEVICE):
        super(CELoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_CE)
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        return self.CELoss(predicted, labels)
