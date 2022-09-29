import torch
import math
import random
import numpy as np
import dgl


def gene_seq(N, p):
    index_lst = []
    for i in range(N):
        for j in range(i):
            index_lst.append([i,j])
    length = len(index_lst)
    num = int(length * p) + 1
    index = torch.tensor(index_lst)
    return index, length, num


def gene_mask(N, index, length, num):
    mask = torch.zeros((N, N), dtype=torch.float)
    random_seq = torch.randperm(length)
    mask_seq = random_seq[:num]
    mask_index = index[mask_seq]
    mask[mask_index[:,0], mask_index[:,1]] = 1.0
    M = mask + mask.T
    return M
    
def gene_affinity1(A, K, t):
    I = torch.eye(A.shape[0], dtype=torch.float)
    d = torch.sum(A, dim=1)
    d_inv = 1 / d
    d_inv[d_inv == np.inf] = 0.0
    D_inv = torch.diag(d_inv)
    A_rw = torch.mm(D_inv, A)
    W = I
    W_temp = I
    for i in range(1, K+1):
        W_temp = torch.mm(A_rw, W_temp)
        W = W + math.exp(-t*i) * W_temp
    return W

def gene_affinity(A, K, t):
    I = torch.eye(A.shape[0], dtype=torch.float)
    d = torch.sum(A, dim=1)
    d_inv = 1 / d
    d_inv[d_inv == np.inf] = 0.0
    D_inv = torch.diag(d_inv)
    A_rw = torch.mm(D_inv, A)
    W = I
    W_temp = I
    for i in range(1, K+1):
        W_temp = torch.mm(A_rw, W_temp)
        W = W + math.exp(-t*i) * W_temp
    d_w = torch.sum(W, dim=1)
    d_w_inv = 1 / d_w
    d_w_inv[d_w_inv == np.inf] = 0.0
    D_w_inv = torch.diag(d_w_inv)
    W_norm = torch.mm(D_w_inv, W)
    return W_norm
    

def mask_feature(feature, mask_prob):
    mask_index = torch.empty(
        (feature.size(1),),
        dtype=torch.float32,
        device=feature.device).uniform_(0, 1) < mask_prob
    aug_feature = feature.clone()
    aug_feature[:, mask_index] = 0
    return aug_feature

def mask_edge(graph, mask_prob):
    num_edges = graph.number_of_edges()
    mask_rates = torch.FloatTensor(np.ones(num_edges) * mask_prob)
    mask = torch.bernoulli(1 - mask_rates)
    mask_idx = mask.nonzero().squeeze(1)
    return mask_idx

def augment_a_graph(graph, x, feat_mask_rate, edge_mask_rate):
    num_node = graph.number_of_nodes()

    edge_mask = mask_edge(graph, edge_mask_rate)
    aug_feature = mask_feature(x, feat_mask_rate)

    aug_structure = dgl.graph([])
    aug_structure.add_nodes(num_node)
    source = graph.edges()[0]
    target = graph.edges()[1]

    aug_source = source[edge_mask]
    aug_target = target[edge_mask]
    aug_structure.add_edges(aug_source, aug_target)
    return aug_structure, aug_feature
 
