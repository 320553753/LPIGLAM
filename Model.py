#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：LPIGLAM 
@File    ：Model.py
@IDE     ：PyCharm 
@Author  ：Shao shaoqi
@Date    ：28/7/2023 下午2:37 
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception135(nn.Module):
    def __init__(self, in_c, c1, c2, c3, kernel):
        super(Inception135, self).__init__()
        self.p1 = nn.Conv1d(in_channels=in_c, out_channels=c1, kernel_size=kernel, dilation=1)
        self.p2 = nn.Conv1d(in_channels=in_c, out_channels=c2, kernel_size=kernel, dilation=3, padding=kernel - 1)
        self.p3 = nn.Conv1d(in_channels=in_c, out_channels=c3, kernel_size=kernel, dilation=5, padding=2 * (kernel - 1))

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        return torch.cat((p1, p2, p3), dim=1)


class Inception123(nn.Module):
    def __init__(self, in_c, c1, c2, c3, kernel):
        super(Inception123, self).__init__()
        self.p1 = nn.Conv1d(in_channels=in_c, out_channels=c1, kernel_size=kernel, dilation=1)
        self.p2 = nn.Conv1d(in_channels=in_c, out_channels=c2, kernel_size=kernel, dilation=2,
                            padding=int((kernel - 1) / 2))
        self.p3 = nn.Conv1d(in_channels=in_c, out_channels=c3, kernel_size=kernel, dilation=3, padding=kernel - 1)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        return torch.cat((p1, p2, p3), dim=1)


class Inception12345(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4, c5, kernel):
        super(Inception12345, self).__init__()
        self.p1 = nn.Conv1d(in_channels=in_c, out_channels=c1, kernel_size=kernel, dilation=1)
        self.p2 = nn.Conv1d(in_channels=in_c, out_channels=c2, kernel_size=kernel, dilation=2,
                            padding=int((kernel - 1) / 2))
        self.p3 = nn.Conv1d(in_channels=in_c, out_channels=c3, kernel_size=kernel, dilation=3, padding=kernel - 1)
        self.p4 = nn.Conv1d(in_channels=in_c, out_channels=c4, kernel_size=kernel, dilation=4,
                            padding=int(3 * (kernel - 1) / 2))
        self.p5 = nn.Conv1d(in_channels=in_c, out_channels=c5, kernel_size=kernel, dilation=5, padding=2 * (kernel - 1))

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        p5 = self.p5(x)

        return torch.cat((p1, p2, p3, p4, p5), dim=1)


class LPIGLAM(nn.Module):
    def __init__(self, hp,
                 protein_MAX_LENGH=1000,
                 lncRNA_MAX_LENGH=2000):
        super(LPIGLAM, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.lncRNA_MAX_LENGTH = lncRNA_MAX_LENGH
        self.lncRNA_kernel = hp.lncRNA_kernel
        self.protein_MAX_LENGTH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel
        self.lncRNA_vocab_size = 5
        self.protein_vocab_size = 21
        self.attention_dim = hp.conv * 4
        self.lncRNA_dim_afterCNNs = self.lncRNA_MAX_LENGTH - self.lncRNA_kernel[0] - self.lncRNA_kernel[1] - self.lncRNA_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3
        self.mix_attention_head = 4

        self.lncRNA_embed = nn.Embedding(
            self.lncRNA_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.lncRNA_CNNs = nn.Sequential(
            Inception123(self.dim, int(self.conv / 2), int(self.conv / 4), int(self.conv / 4), self.lncRNA_kernel[0]),
            nn.ReLU(),
            Inception123(self.conv, self.conv, int(self.conv / 2), int(self.conv / 2), self.lncRNA_kernel[1]),
            nn.ReLU(),
            Inception123(self.conv * 2, self.conv * 2, self.conv, self.conv, self.lncRNA_kernel[2]),
            nn.ReLU(),
        )

        # self.lncRNA_CNNs = nn.Sequential(
        #     Inception135(self.dim, int(self.conv / 2), int(self.conv / 4), int(self.conv / 4), self.lncRNA_kernel[0]),
        #     nn.ReLU(),
        #     Inception135(self.conv, self.conv, int(self.conv / 2), int(self.conv / 2), self.lncRNA_kernel[1]),
        #     nn.ReLU(),
        #     Inception135(self.conv * 2, self.conv * 2, self.conv, self.conv, self.lncRNA_kernel[2]),
        #     nn.ReLU(),
        # )

        # self.lncRNA_CNNs = nn.Sequential(
        #     Inception12345(self.dim, int(self.conv / 5), int(self.conv / 5), int(self.conv / 5), int(self.conv / 5),
        #                    int(self.conv / 5), self.lncRNA_kernel[0]),
        #     nn.ReLU(),
        #     Inception12345(self.conv, int(self.conv / 5 * 2), int(self.conv / 5 * 2), int(self.conv / 5 * 2),
        #                    int(self.conv / 5 * 2), int(self.conv / 5 * 2), self.lncRNA_kernel[1]),
        #     nn.ReLU(),
        #     Inception12345(self.conv * 2, int(self.conv / 5 * 4), int(self.conv / 5 * 4), int(self.conv / 5 * 4),
        #                    int(self.conv / 5 * 4), int(self.conv / 5 * 4), self.lncRNA_kernel[2]),
        #     nn.ReLU(),
        # )


        self.lncRNA_max_pool = nn.MaxPool1d(self.lncRNA_dim_afterCNNs)


        self.Protein_CNNs = nn.Sequential(
            Inception123(self.dim, int(self.conv / 2), int(self.conv / 4), int(self.conv / 4), self.protein_kernel[0]),
            nn.ReLU(),
            Inception123(self.conv, self.conv, int(self.conv / 2), int(self.conv / 2), self.protein_kernel[1]),
            nn.ReLU(),
            Inception123(self.conv * 2, self.conv * 2, self.conv, self.conv, self.protein_kernel[2]),
            nn.ReLU(),
        )

        # self.Protein_CNNs = nn.Sequential(
        #     Inception135(self.dim, int(self.conv / 2), int(self.conv / 4), int(self.conv / 4), self.protein_kernel[0]),
        #     nn.ReLU(),
        #     Inception135(self.conv, self.conv, int(self.conv / 2), int(self.conv / 2), self.protein_kernel[1]),
        #     nn.ReLU(),
        #     Inception135(self.conv * 2, self.conv * 2, self.conv, self.conv, self.protein_kernel[2]),
        #     nn.ReLU(),
        # )

        # self.Protein_CNNs = nn.Sequential(
        #     Inception12345(self.dim, int(self.conv / 5), int(self.conv / 5), int(self.conv / 5), int(self.conv / 5),
        #                    int(self.conv / 5), self.protein_kernel[0]),
        #     nn.ReLU(),
        #     Inception12345(self.conv, int(self.conv / 5 * 2), int(self.conv / 5 * 2), int(self.conv / 5 * 2),
        #                    int(self.conv / 5 * 2), int(self.conv / 5 * 2), self.protein_kernel[1]),
        #     nn.ReLU(),
        #     Inception12345(self.conv * 2, int(self.conv / 5 * 4), int(self.conv / 5 * 4), int(self.conv / 5 * 4),
        #                    int(self.conv / 5 * 4), int(self.conv / 5 * 4), self.protein_kernel[2]),
        #     nn.ReLU(),
        # )


        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)

        self.mix_attention_layer = nn.MultiheadAttention(
            self.attention_dim, self.mix_attention_head)


        self.rnafc1 = nn.Linear(340, 512)
        self.rnafc2 = nn.Linear(512, 160)

        self.profc1 = nn.Linear(399, 512)
        self.profc2 = nn.Linear(512, 160)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(640, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, lncRNA, protein, lncRNAkmer, proteinkmer):
        # [Batchsize, lncRNAseq] -> [Batchsize, lncRNAseq, embed]
        # [Batchsize, proteinseq] -> [Batchsize, proteinseq, embed]
        lncRNAembed = self.lncRNA_embed(lncRNA)
        proteinembed = self.protein_embed(protein)
        # [Batchsize, lncRNAseq, embed] -> [Batchsize, embed, lncRNAseq]
        # [Batchsize, proteinseq, embed] -> [Batchsize, embed, proteinseq]
        lncRNAembed = lncRNAembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # [Batchsize, embed, lncRNAseq] -> [Batchsize, CNNembed, CNNlncRNAseq]
        # [Batchsize, embed, proteinseq]  -> [Batchsize, CNNembed, CNNproteinseq]
        lncRNAConv = self.lncRNA_CNNs(lncRNAembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        # [Batchsize, CNNembed, CNNlncRNAseq] -> [CNNlncRNAseq, Batchsize, CNNembed]
        # [Batchsize, CNNembed, CNNproteinseq] -> [CNNproteinseq, Batchsize, CNNembed]
        lncRNA_QKV = lncRNAConv.permute(2, 0, 1)
        protein_QKV = proteinConv.permute(2, 0, 1)

        # cross Attention
        # [CNNlncRNAseq, Batchsize, CNNembed] -> [CNNlncRNAseq, Batchsize, CNNembed]
        # [CNNproteinseq, Batchsize, CNNembed] -> [CNNproteinseq, Batchsize, CNNembed]
        lncRNA_att, _ = self.mix_attention_layer(lncRNA_QKV, protein_QKV, protein_QKV)
        protein_att, _ = self.mix_attention_layer(protein_QKV, lncRNA_QKV, lncRNA_QKV)

        # [CNNlncRNAseq, Batchsize, CNNembed] -> [Batchsize, CNNembed, CNNlncRNAseq]
        # [CNNproteinseq, Batchsize, CNNembed] -> [Batchsize, CNNembed, CNNproteinseq]
        lncRNA_att = lncRNA_att.permute(1, 2, 0)
        protein_att = protein_att.permute(1, 2, 0)

        lncRNAConv = lncRNAConv * 0.5 + lncRNA_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5

        lncRNAConv = self.lncRNA_max_pool(lncRNAConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)
        lncRNAkmer = self.leaky_relu(self.rnafc2(self.leaky_relu(self.rnafc1(lncRNAkmer))))
        proteinkmer = self.leaky_relu(self.profc2(self.leaky_relu(self.profc1(proteinkmer))))
        pair = torch.cat([lncRNAConv,proteinConv,lncRNAkmer, proteinkmer], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict
