import os.path as osp
import torch
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T
from utils.utils import remove_isolated_nodes


def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
    name = 'dblp' if name == 'DBLP' else name

    if name == 'dblp':
        return CitationFull(path, name, transform=T.NormalizeFeatures())
    else:
        return Planetoid(path, name, transform=T.NormalizeFeatures())



def get_path(base_path, name):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return base_path
    else:
        return osp.join(base_path, name)
    



