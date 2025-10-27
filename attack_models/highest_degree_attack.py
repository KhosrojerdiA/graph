 
import torch
import random
from torch_geometric.utils import degree, remove_self_loops

# promo_mode = False → it removes the neighbor with highest degree (cuts hub connections)

#____________________________________________________________________________________________________________________________________________

def build_highest_degree_attack(data, selected_nodes, budget, promo_mode):

    if isinstance(selected_nodes, torch.Tensor) and selected_nodes.ndimension() == 0:
        selected_nodes = selected_nodes.unsqueeze(0)
    
    selected_nodes = selected_nodes.tolist()
    edge_index = data.edge_index.clone()
    edges = edge_index.t().tolist()
    
    degrees = degree(edge_index[0], num_nodes=data.num_nodes)
    removed_edges_count = 0
    
    while removed_edges_count < budget and edges:
        node = random.choice(selected_nodes)
        node_edges = [(i, (src, dst)) for i, (src, dst) in enumerate(edges) if src == node or dst == node]
        
        if node_edges:
            if promo_mode:
                # Remove edge with the lowest degree node
                edge_to_remove = min(node_edges, key=lambda x: degrees[x[1][1]] if x[1][0] == node else degrees[x[1][0]])
            else:
                # Remove edge with the highest degree node
                edge_to_remove = max(node_edges, key=lambda x: degrees[x[1][1]] if x[1][0] == node else degrees[x[1][0]])
            
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