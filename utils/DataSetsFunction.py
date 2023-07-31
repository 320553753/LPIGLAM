#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：LPIGLAM 
@File    ：DataSetsFunction.py
@IDE     ：PyCharm 
@Author  ：Shao shaoqi
@Date    ：28/7/2023 下午3:36 
'''

import torch
from torch.utils.data import Dataset
import numpy as np

CHARRNASET = {"A": 1, "C": 2, "U": 3, "G": 4}
CHARRNALEN = 4

CHARPROTSET = {"A": 1, "C": 2, "Y": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "V": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                "W": 19,"T": 20}
CHARPROTLEN = 20

rnaelements = 'AUCG'
rnaelement_number = 4

proelements = 'AIYHRDC'
proelement_number = 7
# clusters: {A,G,V}, {I,L,F,P}, {Y,M,T,S}, {H,N,Q,W}, {R,K}, {D,E}, {C}
pro_intab = 'AGVILFPYMTSHNQWRKDEC'
pro_outtab = 'AAAIIIIYYYYHHHHRRDDC'

def label_rna(line, RNA2INT, MAX_RNA_LEN=4000):
    line = line.replace('T', 'U')
    X = np.zeros(MAX_RNA_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_RNA_LEN]):
        X[i] = RNA2INT[ch]
    return X


def label_protein(line, PRO2INT, MAX_PROTEIN_LEN=1000):
    X = np.zeros(MAX_PROTEIN_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_PROTEIN_LEN]):
        X[i] = PRO2INT[ch]
    return X


class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)

def rna_kmer(seq):
    k_mers = ['']
    k_mer_list = []
    k_mer_map = {}
    for T in range(4):
        temp_list = []
        for k_mer in k_mers:
            for x in rnaelements:
                temp_list.append(k_mer + x)
        k_mers = temp_list
        k_mer_list += temp_list

    for i in range(len(k_mer_list)):
        k_mer_map[k_mer_list[i]] = i

    seq = seq.replace('T', 'U')
    seq = ''.join([x for x in seq if x in rnaelements])
    seq_len = len(seq)
    if seq_len == 0:
        return 'Error'
    result = []
    offset = 0
    for K in range(1, 4 + 1):
        vec = [0.0] * (rnaelement_number ** K)
        counter = seq_len - K + 1
        for i in range(seq_len - K + 1):
            k_mer = seq[i:i + K]
            vec[k_mer_map[k_mer] - offset] += 1
        vec = np.array(vec)
        offset += vec.size

        vec = vec / vec.max()
        result += list(vec)
    return np.array(result)

def pro_kmer(seq):
    k_mers = ['']
    k_mer_list = []
    k_mer_map = {}
    for T in range(3):
        temp_list = []
        for k_mer in k_mers:
            for x in proelements:
                temp_list.append(k_mer + x)
        k_mers = temp_list
        k_mer_list += temp_list
    for i in range(len(k_mer_list)):
        k_mer_map[k_mer_list[i]] = i
    transtable = str.maketrans(pro_intab, pro_outtab)
    seq = seq.translate(transtable)
    seq = ''.join([x for x in seq if x in proelements])
    seq_len = len(seq)
    if seq_len == 0:
        return 'Error'
    result = []
    offset = 0
    for K in range(1, 3 + 1):
        vec = [0.0] * (proelement_number ** K)
        counter = seq_len - K + 1
        for i in range(seq_len - K + 1):
            k_mer = seq[i:i + K]
            vec[k_mer_map[k_mer] - offset] += 1
        vec = np.array(vec)
        offset += vec.size

        vec = vec / vec.max()
        result += list(vec)
    return np.array(result)

def collate_fn(batch_data):
    N = len(batch_data)
    rna_ids, protein_ids = [], []
    rna_max = 4000
    protein_max = 1000
    rna_new = torch.zeros((N, rna_max), dtype=torch.long)
    rnakmer_new = torch.zeros((N, 340), dtype=torch.float)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    prokmer_new = torch.zeros((N, 399), dtype=torch.float)
    labels_new = torch.zeros(N, dtype=torch.long)
    for i, pair in enumerate(batch_data):
        pair = pair.strip().split()
        rna_id, protein_id, rnastr, proteinstr, label = pair[-5], pair[-4], pair[-3], pair[-2], pair[-1]

        rna_ids.append(rna_id)
        protein_ids.append(protein_id)
        rnaint = torch.from_numpy(label_rna(
            rnastr, CHARRNASET, rna_max))
        rna_new[i] = rnaint
        rnakmer=torch.from_numpy(rna_kmer(rnastr))
        rnakmer_new[i]=rnakmer
        proteinint = torch.from_numpy(label_protein(
            proteinstr, CHARPROTSET, protein_max))
        protein_new[i] = proteinint
        prokmer = torch.from_numpy(pro_kmer(proteinstr))
        prokmer_new[i] = prokmer
        label = float(label)
        labels_new[i] = np.int(label)
    return (rna_new,rnakmer_new ,protein_new,prokmer_new, labels_new)
