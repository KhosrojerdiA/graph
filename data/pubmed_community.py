import sys
import os

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import VGAE, APPNP
from torch_geometric.utils import negative_sampling, remove_self_loops
import torch_geometric.transforms as T
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score, average_precision_score
import math
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
from torch_geometric.utils import degree, remove_self_loops
import random
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from networkx import pagerank
import pandas as pd
import time
import community.community_louvain as community
from torch_geometric.utils import to_networkx, subgraph


torch.backends.cudnn.deterministic = True  # Force deterministic behavior
torch.backends.cudnn.benchmark = False  # Disable auto-tuning for convolution algorithms
#____________________________________________________________________________________________________________________________


def subraph_pubmed(data, subgraph_nodes, subgraph_edges, dataset_subgraph_path):

    G = to_networkx(data, to_undirected=True)

    # Step 1: Apply Louvain Community Detection
    partition = community.best_partition(G)

    # Step 2: Count Nodes in Each Community
    community_counts = {}
    for node, comm in partition.items():
        community_counts[comm] = community_counts.get(comm, 0) + 1

    # Step 3: Select Communities Until We Reach 4000 Nodes
    selected_nodes = []
    selected_communities = sorted(community_counts.keys(), key=lambda k: -community_counts[k])  # Sort largest first
    
    for comm in selected_communities:
        nodes_in_comm = [node for node, c in partition.items() if c == comm]
        selected_nodes.extend(nodes_in_comm)
        if len(selected_nodes) >= subgraph_nodes:
            break

    # Trim to exactly `subgraph_nodes`
    selected_nodes = selected_nodes[:subgraph_nodes]
    selected_nodes = torch.tensor(selected_nodes, dtype=torch.long)

    # Step 4: Extract Induced Subgraph
    sub_edge_index, _ = subgraph(selected_nodes, data.edge_index, relabel_nodes=True)

    # Step 5: Ensure Exactly 12000 Edges
    if sub_edge_index.shape[1] > subgraph_edges:
        selected_edges = torch.randperm(sub_edge_index.shape[1])[:subgraph_edges]
        sub_edge_index = sub_edge_index[:, selected_edges]

    # Step 6: Update Features, Labels, and Masks
    new_x = data.x[selected_nodes]
    new_y = data.y[selected_nodes]

    new_train_mask = data.train_mask[selected_nodes]
    new_val_mask = data.val_mask[selected_nodes]
    new_test_mask = data.test_mask[selected_nodes]

    # Step 7: Create the new subgraph data object
    sub_data = data.clone()
    sub_data.x = new_x
    sub_data.y = new_y
    sub_data.edge_index = sub_edge_index
    sub_data.train_mask = new_train_mask
    sub_data.val_mask = new_val_mask
    sub_data.test_mask = new_test_mask

    torch.save(sub_data, dataset_subgraph_path)
    print(f"Subgraph saved to {dataset_subgraph_path}")

#____________________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________________

main_seed = 1147
torch.manual_seed(main_seed)
torch.cuda.manual_seed(main_seed)
np.random.seed(main_seed)
random.seed(main_seed)

dataset_subgraph_path = "/mnt/data/khosro/Graph-Representation-Learning-for-Strategic-Edge-Removal-in-Keyword-Search-Demotion-Attacks/data/pubmed_subgraph.pt"
subgraph_nodes = 6000
subgraph_edges = 20000

dataset = Planetoid(root='data/Planetoid', name='PubMed')
data = dataset[0]
subraph_pubmed(data, subgraph_nodes, subgraph_edges, dataset_subgraph_path)
loaded_dataset_subgraph = torch.load(dataset_subgraph_path)



#____________________________________________________________________________________________________________________________
