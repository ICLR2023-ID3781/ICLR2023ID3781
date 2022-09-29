import numpy as np
import torch
import dgl


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