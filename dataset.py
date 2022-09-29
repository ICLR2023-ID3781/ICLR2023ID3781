import torch
import numpy as np
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, CoauthorCSDataset, CoauthorPhysicsDataset


def load_dataset(name_dataset):
    if name_dataset == 'cora':
        dataset = CoraGraphDataset()
    elif name_dataset == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name_dataset == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name_dataset == 'cs':
        dataset = CoauthorCSDataset()
    elif name_dataset == 'physics':
        dataset = CoauthorPhysicsDataset()
    elif name_dataset == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name_dataset == 'computer':
        dataset = AmazonCoBuyComputerDataset()
    

    graph = dataset[0]
    datasets_public_split = ['cora', 'citeseer', 'pubmed']
    datasets_personal_split = ['photo', 'computer', 'cs', 'physics']

    if name_dataset in datasets_public_split:
        # use public split
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()

    if name_dataset in datasets_personal_split:
        # use 1/1/8 split for train/validation/test
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        N = graph.number_of_nodes()
        train_num = int(N * train_ratio)
        val_num = int(N * (train_ratio + val_ratio))

        idx = np.arange(N)
        np.random.shuffle(idx)

        train_idx = idx[:train_num]
        val_idx = idx[train_num:val_num]
        test_idx = idx[val_num:]

        train_idx = torch.tensor(train_idx)
        val_idx = torch.tensor(val_idx)
        test_idx = torch.tensor(test_idx)

    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')

    return graph, feat, labels, num_class, train_idx, val_idx, test_idx
