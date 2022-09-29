# Matrix Correlation
import argparse
import sys
from model import Pipeline, LogReg
from dataset import load_dataset
import numpy as np
import torch
import torch.nn as nn
from utils import *
import warnings
import time
import os.path as osp
import os
import sys

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Matrix Correlation')

parser.add_argument('--name_dataset', type=str, default='pubmed', help='Dataset for train and test.')
parser.add_argument('--gpu_index', type=int, default=0, help='Choosen GPU index.')
parser.add_argument('--num_epochs', type=int, default=100, help='Epochs for training model.')
parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate of GCN backbone.')
parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of logistic regression model.')
parser.add_argument('--wd1', type=float, default=0, help='Weight decay of GCN backbone.')
parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay of logistic regression model.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN of MLP layers')
parser.add_argument('--use_mlp', action='store_true', default=False, help='Use MLP as backbone')
parser.add_argument('--pe', type=float, default=0.5, help='Drop ratio for edge.')
parser.add_argument('--pf', type=float, default=0.2, help='Drop ratio for node feature.')
parser.add_argument("--hid_dim", type=int, default=512, help='Hidden dimension.')
parser.add_argument("--out_dim", type=int, default=512, help='Output dimemsion.')
parser.add_argument('--alpha', type=float, default=20, help='Coefficient for statistical metrics.')
parser.add_argument('--beta', type=float, default=7, help='Coefficient for dimension decorrelation.')

args = parser.parse_args()



if __name__ == '__main__':
    # choose gpu or cpu
    if args.gpu_index >= 0 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu_index)
    else:
        args.device = 'cpu'
    print("super-parameters are as follows:")
    print(args)
    graph, feature, labels, num_classes, train_idx, val_idx, test_idx = load_dataset(args.name_dataset)
    in_dim = feature.shape[1]

    model = Pipeline(in_dim, args.hid_dim, args.out_dim, args.num_layers, args.use_mlp)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    N = graph.number_of_nodes()
    
    I = torch.eye(N, dtype=torch.float)
    adj = graph.adjacency_matrix().to_dense() + I
    d = torch.sum(adj, dim=1)
    d_inv = 1 / d
    d_inv[d_inv == np.inf] = 0.0
    D_inv = torch.diag(d_inv)
    D = torch.diag(d)
    # print("max degree", d.max())

    
    I_D = torch.eye(args.out_dim, dtype=torch.float).to(args.device)
    M = torch.ones((args.out_dim, args.out_dim), dtype=torch.float).to(args.device) - I_D

    M_N = torch.ones((N, N), dtype=torch.float).to(args.device) - torch.eye(N, dtype=torch.float).to(args.device)
    
    start_time = time.time()
    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()

        graph1, feature1 = augment_a_graph(graph, feature, args.pf, args.pe)
        graph2, feature2 = augment_a_graph(graph, feature, args.pf, args.pe)

        graph1 = graph1.add_self_loop()
        graph2 = graph2.add_self_loop()

        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)

        feature1 = feature1.to(args.device)
        feature2 = feature2.to(args.device)

        h1, h2 = model(graph1, feature1, graph2, feature2)
        
        numerator = torch.trace(h1.T @ h2)   
        denominator = torch.sqrt(torch.trace(h1.T @ h1) * torch.trace(h2.T @ h2))
        MC_c = numerator / denominator
        loss_cons = - MC_c

        correlation1 = torch.mm(h1.T, h1)
        correlation2 = torch.mm(h2.T, h2)
        correlation3 = torch.mm(h1.T, h2)
        correlation1 = correlation1 / N
        correlation2 = correlation2 / N
        correlation3 = correlation3 / N
        loss_intra_dec1 = (correlation1 * M).pow(2).sum()
        loss_intra_dec2 = (correlation2 * M).pow(2).sum()
        loss_inter_dec = (correlation3 * M).pow(2).sum()
        loss_dec = (loss_intra_dec1 + loss_intra_dec2 + loss_inter_dec) / args.out_dim / (args.out_dim - 1)

        loss = args.alpha * loss_cons + args.beta * loss_dec
        
        loss.backward()
        optimizer.step()
        
        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))
            
    end_time = time.time()
    

    print("======== Evaluation =========")
    graph = graph.to(args.device)
    graph = graph.remove_self_loop().add_self_loop()
    feature = feature.to(args.device)

    embeds = model.get_embedding(graph, feature)

    train_embs = embeds[train_idx]
    val_embs = embeds[val_idx]
    test_embs = embeds[test_idx]

    label = labels.to(args.device)

    train_labels = label[train_idx]
    val_labels = label[val_idx]
    test_labels = label[test_idx]
    
    
    ''' Linear Evaluation '''
    logreg = LogReg(train_embs.shape[1], num_classes)
    opt = torch.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

    logreg = logreg.to(args.device)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0

    for epoch in range(2000):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = torch.argmax(logits, dim=1)
        train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with torch.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)

            val_preds = torch.argmax(val_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)

            val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                if test_acc > eval_acc:
                    eval_acc = test_acc

            # print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc))
    print('Linear evaluation accuracy:{:.4f}'.format(eval_acc))
    
