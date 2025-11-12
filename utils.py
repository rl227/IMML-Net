import numpy as np
import scipy.io as scio
from sklearn.neighbors import kneighbors_graph
import torch
import scipy.sparse as sp
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from clusteringPerformence import similarity_function, StatisticClustering
import numpy.linalg as LA
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.preprocessing import normalize

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def features_to_adj(path="./data/", datasets="scene15"):
    print("loading {} data...".format(datasets))
    data = scio.loadmat("{}{}.mat".format(path, datasets))
    x = data["X"]
    adj = []
    features = []
    count = 0
    for i in range(x.shape[1]):
        features.append(normalize(x[0, i]))
        temp = kneighbors_graph(features[i], 10)
        temp = sp.coo_matrix(temp)
        temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
        temp = normalize(temp + sp.eye(temp.shape[0]))
        temp = sparse_mx_to_torch_sparse_tensor(temp)
        adj.append(temp)
        count += 1
    labels = data["Y"]
    labels = labels.reshape(-1, )
    return adj, features, labels, count

def features_to_adj_1(path="./data/", datasets="ACM"):
    print("loading {} data...".format(datasets))
    data = scio.loadmat("{}{}.mat".format(path, datasets))
    x = data["X"]
    adj = []
    features = []
    count = 0
    for i in range(x.shape[1]):
        features.append(x[0, i])
        count += 1
    labels = data["Y"]
    A1 = data["PAP"]
    adj.append(A1)
    A2 = data["PLP"]
    adj.append(A2)
    A3 = data["PMP"]
    adj.append(A3)
    A4 = data["PTP"]
    adj.append(A4)
    labels = labels.reshape(-1, )
    return adj, features, labels, count

def features_to_adj_2(path="./data/", datasets="DBLP"):
    print("loading {} data...".format(datasets))
    data = scio.loadmat("{}{}.mat".format(path, datasets))
    x = data["X"]
    adj = []
    features = []
    count = 0
    for i in range(x.shape[1]):
        features.append(x[0, i])
        count += 1
    labels = data["Y"]
    A1 = data["net_APA"]
    adj.append(A1)
    A2 = data["net_APCPA"]
    adj.append(A2)
    A3 = data["net_APTPA"]
    adj.append(A3)
    labels = labels.reshape(-1, )
    return adj, features, labels, count

def features_to_adj_3(path="./data/", datasets="IMDB"):
    print("loading {} data...".format(datasets))
    data = scio.loadmat("{}{}.mat".format(path, datasets))
    x = data["X"]
    adj = []
    features = []
    count = 0
    for i in range(x.shape[1]):
        features.append(x[0, i])
        count += 1
    labels = data["Y"]
    A1 = data["MAM"]
    adj.append(A1)
    A2 = data["MDM"]
    adj.append(A2)
    A3 = data["MYM"]
    adj.append(A3)
    labels = labels.reshape(-1, )
    return adj, features, labels, count

def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j

def calc_iic_loss(x_out, x_tf_out, lamb=1.0, EPS=1e-10):
    # has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) \
                      - lamb * torch.log(p_j) \
                      - lamb * torch.log(p_i))

    loss = loss.sum()
    return loss

def mask_correlated_samples(n):
    N = 2 * n
    # N =  batch_size
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(n):
        mask[i, n + i] = 0
        mask[n + i, i] = 0
    mask = mask.bool()
    return mask

def C_loss(Z1, Z2, n):
    mask = mask_correlated_samples(n)
    criterion_c = nn.CrossEntropyLoss(reduction="mean")
    similarity_f = nn.CosineSimilarity(dim=2)
    N = 2 * n
    # N = self.batch_size
    z = torch.cat((Z1, Z2), dim=0)

    sim = torch.matmul(z, z.T) / 0.5
    # sim = similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
    sim_i_j = torch.diag(sim, n)
    sim_j_i = torch.diag(sim, -n)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    # print(logits.shape, labels.shape)
    loss = criterion_c(logits, labels)
    loss /= N

    return loss

def weights_init(m):
    if isinstance(m, (nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def plot(epoch, per_list, loss_list):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    dx1 = [t for t in range(0, epoch)]
    ax1.set_ylabel('Objective Value/ACC', size=15)
    ax1.set_xlabel('Epoch', size=15)
    p1 = ax1.plot(dx1, per_list, label='ACC', linewidth=2, c='b')
    p2 = ax2.plot(dx1, loss_list, label='Loss', linewidth=2, c='r')
    line = p1 + p2
    ax1.legend(line, [l.get_label() for l in line], loc='best')
    plt.grid(color="k", linestyle="--", linewidth=0.5, axis='y')
    plt.xlim((0, epoch))
    # plt.savefig('./dataO3/FZ11/' + str(data_name_plot) + '.svg')
    plt.show()

def plot1(per_list1, true_label1):
    dx = [t for t in range(0, 2881)]
    plt.ylabel('Concentration', size=15)
    plt.xlabel('Time', size=15)
    plt.plot(dx, per_list1, label='Prediction Label', linewidth=1, c='b')
    plt.plot(dx, true_label1, label='True Label', linewidth=1, c='r')
    plt.legend(loc='best')
    plt.grid(color="k", linestyle="--", linewidth=0.5)
    plt.xlim((0, 240))
    # plt.savefig('./dataO3/FZGZ/' + str(data_name_plot) + '_' + str(epoch) + '.svg')
    plt.show()

def plot_tsne(array, gnd):
    X_tsne = TSNE(n_components=2, learning_rate=100,  random_state=0).fit_transform(array)
    plt.figure(figsize=(12, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = gnd)
    # plt.colorbar()
    plt.show()

def plot_hot_picture(W):
    sns.set()
    np.random.seed(0)
    u1 = np.array(W)
    # sns.heatmap(u1, cmap="Greys")
    plt.axis('off')
    sns.heatmap(W, annot=True)
    # plt.savefig("MNIST.jpg")
    plt.show()

def spectral_clustering(points, k, gnd, repnum=10):
    # W = similarity_function(points)
    W = points
    Dn = np.diag(1 / np.sqrt(np.sum(W, axis=1)))
    L = np.dot(np.dot(Dn, W), Dn)
    L[np.isnan(L)] = 0
    eigvals, eigvecs = LA.eig(L)
    eigvecs = eigvecs.astype(float)
    indices = np.argsort(eigvals)[:k]
    k_smallest_eigenvectors = normalize(eigvecs[:, indices])

    [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(k_smallest_eigenvectors, gnd, k)
    print("ACC, NMI, ARI, Purity, Fscore, Precision, Recall: ", ACC, NMI, ARI, Purity, Fscore, Precision, Recall)
    return [ACC, NMI, Purity, ARI, Fscore, Precision, Recall]

if __name__ == '__main__':
    adj, features, labels = features_to_adj()
    # print(adj)
    print(features[3])
    print(features[3].shape)
    print(len(features[3][1]))
