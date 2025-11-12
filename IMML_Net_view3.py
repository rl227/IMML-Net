from __future__ import print_function, division
import argparse
import random
import time
from clusteringPerformence import StatisticClustering
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.parameter import Parameter
from torch.nn import Linear
from GCN_Fuse_model import GCN, CombineNet
from utils import features_to_adj
from DataStardardization import normalization, standardization, zero_score
from LoadF import loadFW, loadGW, loadMFW, loadMGW
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from layers import GraphConvolution


class RSSLRNet(nn.Module):
    def __init__(self, features, adj, Y1, Y2, u0, umax, p0, e1, e2, y1, n, n_cluster, dv, blocks, para, use_cuda):
        super(RSSLRNet, self).__init__()
        self.use_cuda = use_cuda
        if self.use_cuda:
            print("GPU activated")

        self.u0 = nn.Parameter(torch.FloatTensor([u0]), requires_grad=True)
        self.umax = umax
        self.p0 = p0
        self.e1 = e1
        self.e2 = e2
        self.y1 = y1
        self.para = para

        self.n = n
        self.c = n_cluster
        self.dv = dv
        self.blocks = blocks
        self.use_cuda = use_cuda
        self.active_para = nn.Parameter(torch.FloatTensor([self.para]), requires_grad=True)
        self.active_para1 = nn.Parameter(torch.FloatTensor([self.para]), requires_grad=True)
        self.active_para2 = nn.Parameter(torch.FloatTensor([self.para]), requires_grad=True)


        self.X = features
        self.adj = adj
        self.Y1 = Y1
        self.Y2 = Y2

        self.con1 = nn.Linear(self.n, self.n, bias=False)
        self.con2 = nn.Linear(self.n, self.n, bias=False)
        self.lag1 = nn.Linear(self.n, self.n, bias=False)
        self.lag2 = nn.Linear(self.n, self.n, bias=False)
        self.lag3 = nn.Linear(self.n, self.n, bias=False)
        self.lag = nn.Linear(self.n, self.n, bias=False)
        self.bn_input_01 = nn.BatchNorm1d(self.n, momentum=0.6)

    def self_active_svd(self, x, thershold):
        return F.leaky_relu(x - 1 / thershold)

    def self_active_sh(self, x, thershold):
        return F.leaky_relu(x - 1 / thershold) - F.leaky_relu(-1.0 * x - 1 / thershold)

    def self_active_l21(self, x, thershold):
        nw = torch.norm(x)
        if nw > thershold:
            x = F.selu(nw - thershold) * x / nw
        else:
            x = torch.zeros_like(x)
        return x

    def forward(self, X, adj):

        Z = list()
        W = list()

        E1 = []
        E2 = []
        E3 = []

        Y1_1 = []
        Y1_2 = []
        Y1_3 = []

        Y2 = list()
        uk = list()

        Z_init = self.con2(self.bn_input_01(adj[0] + adj[1] + adj[2]))
        Z.append(self.self_active_sh(Z_init, self.active_para1))

        # init J
        u, s, v = torch.svd(Z[-1])
        s6 = self.self_active_svd(s, self.active_para)
        s6 = torch.diag(s6)
        W.append(F.selu(u.mm(s6).mm(v)))
        print(self.active_para)

        # init E
        E_init_tmp_1 = X[0] - X[0].mm(Z[-1])
        # E1.append(self.self_active_sh(E_init_tmp_1, self.active_para2))
        E_init_tmp_1_zero = torch.zeros_like(E_init_tmp_1)
        for j in range(E_init_tmp_1.size(1)):
            E_init_tmp_tmp_1 = E_init_tmp_1[:, j]
            E_init_tmp_1_zero[:, j] = self.self_active_l21(E_init_tmp_tmp_1, self.active_para2)
        E1.append(E_init_tmp_1_zero)

        E_init_tmp_2 = X[1] - X[1].mm(Z[-1])
        # E2.append(self.self_active_sh(E_init_tmp_2, self.active_para2))
        E_init_tmp_2_zero = torch.zeros_like(E_init_tmp_2)
        for j in range(E_init_tmp_2.size(1)):
            E_init_tmp_tmp_2 = E_init_tmp_2[:, j]
            E_init_tmp_2_zero[:, j] = self.self_active_l21(E_init_tmp_tmp_2, self.active_para2)
        E2.append(E_init_tmp_2_zero)

        E_init_tmp_3 = X[2] - X[2].mm(Z[-1])
        # E3.append(self.self_active_sh(E_init_tmp_3, self.active_para2))
        E_init_tmp_3_zero = torch.zeros_like(E_init_tmp_3)
        for j in range(E_init_tmp_3.size(1)):
            E_init_tmp_tmp_3 = E_init_tmp_3[:, j]
            E_init_tmp_3_zero[:, j] = self.self_active_l21(E_init_tmp_tmp_3, self.active_para2)
        E3.append(E_init_tmp_3_zero)

        # init Y1, Y2
        Y1_1.append(self.lag1(self.Y1[0] + self.u0 * X[0]))
        Y1_2.append(self.lag2(self.Y1[1] + self.u0 * X[1]))
        Y1_3.append(self.lag3(self.Y1[2] + self.u0 * X[2]))
        Y2.append(self.lag(self.Y2 + self.u0 * Z[-1]))
        uk.append(self.u0)

        for k in range(self.blocks):

            Z_tmp = self.con1(self.bn_input_01(Z[-1])) + self.con2(self.bn_input_01(X[0].t().mm(X[0]) - X[0].t().mm(E1[-1]) + X[0].t().mm(Y1_1[-1] / uk[-1]) +
                     X[1].t().mm(X[1]) - X[1].t().mm(E2[-1]) + X[1].t().mm(Y1_2[-1] / uk[-1]) +
                     X[2].t().mm(X[2]) - X[2].t().mm(E3[-1]) + X[2].t().mm(Y1_3[-1] / uk[-1]) + (W[-1] - Y2[-1] / uk[-1])))
            Z.append(self.self_active_sh(Z_tmp, self.active_para1))

            # Update J
            Z_w_tmp = Z[-1] + Y2[-1] / uk[-1]
            u1, s1, v1 = torch.svd(Z_w_tmp)
            s2 = self.self_active_svd(s1, self.active_para)
            s2 = torch.diag(s2)
            W.append(F.selu(u1.mm(s2).mm(v1)))
            # print(W[-1])

            # Update E
            E_tmp_tmp_1 = X[0] - X[0].mm(Z[-1]) + Y1_1[-1] / uk[-1]
            E_tmp_tmp_1_zero = torch.zeros_like(E_tmp_tmp_1)
            for j in range(E_tmp_tmp_1.size(1)):
                E_tmp_tmp_tmp_1 = E_tmp_tmp_1[:, j]
                E_tmp_tmp_1_zero[:, j] = self.self_active_l21(E_tmp_tmp_tmp_1, self.active_para2)
            E1.append(E_tmp_tmp_1_zero)

            E_tmp_tmp_2 = X[1] - X[1].mm(Z[-1]) + Y1_2[-1] / uk[-1]
            E_tmp_tmp_2_zero = torch.zeros_like(E_tmp_tmp_2)
            for j in range(E_tmp_tmp_2.size(1)):
                E_tmp_tmp_tmp_2 = E_tmp_tmp_2[:, j]
                E_tmp_tmp_2_zero[:, j] = self.self_active_l21(E_tmp_tmp_tmp_2, self.active_para2)
            E2.append(E_tmp_tmp_2_zero)

            E_tmp_tmp_3 = X[2] - X[2].mm(Z[-1]) + Y1_3[-1] / uk[-1]
            E_tmp_tmp_3_zero = torch.zeros_like(E_tmp_tmp_3)
            for j in range(E_tmp_tmp_3.size(1)):
                E_tmp_tmp_tmp_3 = E_tmp_tmp_3[:, j]
                E_tmp_tmp_3_zero[:, j] = self.self_active_l21(E_tmp_tmp_tmp_3, self.active_para2)
            E3.append(E_tmp_tmp_3_zero)

            # Update Y1, Y2
            Y1_1.append(self.lag1(Y1_1[-1] + uk[-1] * (X[0] - X[0].mm(Z[-1]) - E1[-1])))
            Y1_2.append(self.lag2(Y1_2[-1] + uk[-1] * (X[1] - X[1].mm(Z[-1]) - E2[-1])))
            Y1_3.append(self.lag3(Y1_3[-1] + uk[-1] * (X[2] - X[2].mm(Z[-1]) - E3[-1])))
            Y2.append(self.lag(Y2[-1] + uk[-1] * (Z[-1] - W[-1])))

        return Z


