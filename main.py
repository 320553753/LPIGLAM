#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：LPIGLAM 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：Shao shaoqi
@Date    ：28/7/2023 下午2:36 
'''
import argparse

from RunModel import run_model
from model import LPIGLAM

parser = argparse.ArgumentParser(
    prog='LPIGLAM',
    description='LPIGLAM is model in paper: \"Predicting Plant LncRNA-Protein Interactions through Global and Local features based on Attention Mechanism\"',
    epilog='Model config set by config.py')

parser.add_argument('-d','--dataSetName', default="ATH",choices=[
                    "ATH", "ZEA", "RPI369","RPI488","RPI1807","RPI2241","NPInter"], help='Enter which dataset to use for the experiment')
parser.add_argument('-m', '--model', choices=['LPIGLAM'],
                    default='LPIGLAM', help='Which model to use, \"LPIGLAM\" is used by default')
parser.add_argument('-s', '--seed', type=int, default=114514,
                    help='Set the random seed, the default is 114514')
parser.add_argument('-f', '--fold', type=int, default=5,
                    help='Set the K-Fold number, the default is 5')
args = parser.parse_args()

if args.model == 'LPIGLAM':
    run_model(SEED=args.seed, DATASET=args.dataSetName,
              MODEL=LPIGLAM, K_Fold=args.fold, LOSS='CrossEntropy')

