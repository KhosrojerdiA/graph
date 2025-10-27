 
import torch
import random
from torch_geometric.utils import remove_self_loops
import numpy as np

#____________________________________________________________________________________________________________________________________________

def build_random_attack(data, selected_nodes, budget, promo_mode):

    torch.seed()
    np.random.seed(None)
    random.seed()

    if isinstance(selected_nodes, torch.Tensor) and selected_nodes.ndimension() == 0:
        selected_nodes = selected_nodes.unsqueeze(0)
    
    selected_nodes = selected_nodes.tolist()
    edge_index = data.edge_index.clone()
    edges = edge_index.t().tolist()
    
    removed_edges_count = 0
    
    while removed_edges_count < budget and edges:
        node = random.choice(selected_nodes)  # Select a random node from selected_nodes
        node_edges = [(i, (src, dst)) for i, (src, dst) in enumerate(edges) if src == node or dst == node]
        
        if node_edges:
            edge_to_remove = random.choice(node_edges)
            edges.pop(edge_to_remove[0])
            removed_edges_count += 1
    
    # Create a new edge_index tensor from the modified edges
    new_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    new_edge_index, _ = remove_self_loops(new_edge_index)
    
    # Assign the modified edge_index to create the updated dataset
    updated_data = data.clone()
    updated_data.edge_index = new_edge_index
    
    return updated_data

#____________________________________________________________________________________________________________________________________________