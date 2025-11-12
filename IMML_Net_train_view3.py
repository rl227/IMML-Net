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
# from LADMAP_Net import LADMAPNet
# from LADMAP_Net_view6 import LADMAPNet
from IMML_Net_view3 import RSSLRNet
from utils import features_to_adj, weights_init, plot, plot1, spectral_clustering, plot_tsne, plot_hot_picture
from DataStardardization import normalization, standardization, zero_score
from LoadF import loadFW, loadGW, loadMFW, loadMGW
from LoadF import loadFW, loadGW, loadMFW, loadMGW, loadMGW1, loadMGW2, loadMGW3, loadMGW4, loadMGW5, loadMGW6
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchviz import make_dot
import sys
import scipy
sys.setrecursionlimit(2000)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def train(model, epoch, lr, alpha, beta, n_cluster):
    acc_max = 0.0
    loss_Self1 = 0.0
    loss_Self2 = 0.0
    best_ACC = []
    per_list = []
    loss_list = []
    per_ = []
    # optimizer = optim.Adam(model.parameters(), lr=lr) if epoch < 20 else optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=15, verbose=True,
                                                           min_lr=1e-8)
    criterion = nn.MSELoss()
    criterion_Smooth = nn.SmoothL1Loss()
    criterion_KL = nn.KLDivLoss()
    criterion_BCE = nn.BCEWithLogitsLoss()
    I = torch.eye(features[0].shape[1], features[0].shape[1]).float().cuda()

    for i in range(epoch):
        model.train()
        optimizer.zero_grad()
        output_h = model(features, adj)
        output_h = output_h[-1]
        # print(output_h)


        loss_Self1 = criterion(output_h, features[0].t().mm(features[0])) + criterion(output_h, features[1].t().mm(features[1])) + criterion(output_h, features[2].t().mm(features[2]))
        loss_Self2 = criterion(output_h, adj[0]) + criterion(output_h, adj[1]) + criterion(output_h, adj[2])
        loss_Self1 = alpha * loss_Self1
        loss_Self2 = alpha * loss_Self2
        loss_Ori = beta * (criterion(output_h.t().mm(output_h), I))
        loss = loss_Self1 + loss_Self2 + loss_Ori

        # make_dot(loss, params=dict(model.named_parameters())).render(f"graph", format="png")
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # AE_loss = loss_AE.cpu().detach().numpy()
        # Cla_loss = loss_Cla.cpu().detach().numpy()
        # Con_loss = loss_Con.cpu().detach().numpy()
        Self1_loss = loss_Self1.cpu().detach().numpy()
        Self2_loss = loss_Self2.cpu().detach().numpy()
        Ori_loss = loss_Ori.cpu().detach().numpy()
        train_loss = loss.cpu().detach().numpy()
        loss_list.append(train_loss)

        # print("epoch", i, "AE loss:{:.16f}".format(AE_loss))
        # print("epoch", i, "Cla loss:{:.16f}".format(Cla_loss))
        # print("epoch", i, "Con loss:{:.16f}".format(Con_loss))
        print("epoch", i, "Self1 loss:{:.16f}".format(Self1_loss))
        print("epoch", i, "Self2 loss:{:.16f}".format(Self2_loss))
        print("epoch", i, "Ori loss:{:.16f}".format(Ori_loss))
        print("epoch", i, "loss:{:.16f}".format(train_loss))

        output_hh = output_h.cpu().detach().numpy()
        # scipy.io.savemat('Notting-Hill_Z.mat', mdict={'output_hh': output_hh})
        # [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = spectral_clustering((output_hh), n_cluster, labels, repnum=10)
        [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(output_hh, labels, n_cluster)
        print(ACC, NMI, Purity, ARI, Fscore, Precision, Recall)
        per_list.append(ACC[0]*100)
        per_.append([ACC, NMI, Purity, ARI, Fscore, Precision, Recall])
        if (ACC[0] > acc_max):
            acc_max = ACC[0]
            epoch_best_mlp = i
    print("acc_max", np.around(acc_max, decimals=4))
    print("epoch_best", np.around(epoch_best_mlp, decimals=4))
    # plot(epoch, per_list, loss_list)
    # plot_tsne(output_hh, labels)
    # plot_hot_picture(output_hh)
    # sio.savemat('./per_loss1/scene15_clustering_per_.mat', {'idx': per_})
    # sio.savemat('./per_loss1/scene15_clustering_loss_total.mat', {'idx': loss_list})

    alpha_str = str(alpha).replace('.', '_')
    beta_str = str(beta).replace('.', '_')

    sio.savemat(f'./per_loss1/scene15_1_clustering_per_alpha{alpha_str}_beta{beta_str}.mat', {'idx': per_})
    sio.savemat(f'./per_loss1/scene15_1_clustering_loss_total_alpha{alpha_str}_beta{beta_str}.mat',
                {'idx': loss_list})

if __name__ == '__main__':
    alpha_list = [10.0]
    beta_list = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    # alpha_list = [0.0001]
    # beta_list = [0.1, 1.0, 10.0]
    # alpha_list = [1.0]
    # beta_list = [1.0]
    for ii in range(len(alpha_list)):
        for j in range(len(beta_list)):
            print("---------------------------------------------------------------")
            print("This is i", ii, "This is j", j)
            seed = 2022
            torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # use_cuda = False
            use_cuda = torch.cuda.is_available()
            datasetW_dir = "./datasetW"  # 所有视角的相似性矩阵
            adj = []
            dv = []
            Y1 = []

            adj_tmp = loadMGW(os.path.join(datasetW_dir, "scene15MGW.mat"))  # 单个视角拉普拉斯矩阵LG
            adj.append(adj_tmp[0][0])  # - 0.2
            adj.append(adj_tmp[0][1])  # - 0.2
            adj.append(adj_tmp[0][2])  # - 0.2

            adj1, features, labels, view_count = features_to_adj()

            adj_all = adj[0] + adj[1] + adj[2]
            n_cluster = len(set(np.array(labels)))
            [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = spectral_clustering(adj_all, n_cluster, labels, repnum=10)
            [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(adj_all, labels, n_cluster)
            print(ACC, NMI, Purity, ARI, Fscore, Precision, Recall)

            for i in range(len(features)):

                features_0 = features[i].T / 1.0
                features_0 = torch.from_numpy(features_0).float()
                # features_0 = normalization(features_0)  # NUS-WIDE注释
                features_0 = standardization(features_0)  # NUS-WIDE注释
                # features_0 = zero_score(features_0)
                features[i] = features_0.cuda()
                # adj_0 = adj[i].to_dense().float()
                adj_0 = torch.from_numpy(adj[i]).float()
                adj[i] = adj_0.cuda()
                dv.append(features[i].size(0))
                Y1.append(torch.zeros_like(features[i]).cuda())

            print(dv)
            n = len(features[1][0])
            Y2 = torch.zeros(n, n).cuda()
            n_cluster = len(set(np.array(labels)))
            u0 = 10e-1
            umax = 10e10
            p0 = 1.1
            e1 = 10e-6
            e2 = 10e-2
            y1 = torch.norm(features[1], 2)
            print(y1)

            blocks = 1
            lr = 0.001
            epoch = 200
            para = 0.1
            # model = LADMAPNet(features, adj, Y1, Y2, u0, umax, p0, e1, e2, y1, n, n_cluster, dv, blocks, para, use_cuda).cuda()  # Caltech101-all 20 0.1 0.005
            model = RSSLRNet(features, adj, Y1, Y2, u0, umax, p0, e1, e2, y1, n, n_cluster, dv, blocks, para, use_cuda).cuda()  # Caltech101-all 20 0.1 0.005
            # model.apply(weights_init)

            alpha = alpha_list[ii]
            beta = beta_list[j]
            print(alpha, beta)
            train(model, epoch, lr, alpha, beta, n_cluster)

