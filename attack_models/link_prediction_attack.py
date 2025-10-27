
import torch
from torch_geometric.utils import remove_self_loops
from torch_geometric.transforms import NormalizeFeatures
from utils.utils import *

#____________________________________________________________________________________________________________________________________________

def create_test_set_rlp(data, selected_nodes):

    mask = torch.isin(data.edge_index[0], selected_nodes) | torch.isin(data.edge_index[1], selected_nodes)
    filtered_edge_index = data.edge_index[:, mask]
    test_set = data.clone()
    test_set.edge_index = filtered_edge_index

    return test_set

#____________________________________________________________________________________________________________________________________________

def predict_edges_rlp(model, test_set, data, n_edges, promo_mode):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_set = NormalizeFeatures()(test_set).to(device)
    data = NormalizeFeatures()(data).to(device)
    
    model.eval()
    with torch.no_grad():
        z = model.encode(test_set.x, test_set.edge_index)
        edge_scores = model.decoder(z, test_set.edge_index)
    
    # Select edges based on pruning mode
    if promo_mode:
        sorted_indices = torch.argsort(edge_scores, descending=True)[:n_edges]  # Highest scores
    else:
        sorted_indices = torch.argsort(edge_scores)[:n_edges]  # Lowest scores
    
    if n_edges == 1:
        sorted_indices = sorted_indices[0].unsqueeze(0)  # Ensure a single edge is handled correctly
    
    # Find the corresponding edges in test_set
    edges_to_remove = test_set.edge_index[:, sorted_indices]
    
    # Convert edges_to_remove to a list of tuples
    edges_to_remove_list = list(map(tuple, edges_to_remove.t().tolist()))  # Format [(node1, node2), ...]

    return edges_to_remove_list 



def predict_edges_vgae(model, z, data, node_id):
    """
    Use VGAE decoder (inner product) to score edges touching node_id.
    Returns { (u,v): score } where score ~ link prob (sigmoid(inner product)).
    """
    edge_index = data.edge_index
    mask = (edge_index[0] == node_id) | (edge_index[1] == node_id)
    connected_edges = edge_index[:, mask]

    preds = {}
    with torch.no_grad():
        for i in range(connected_edges.size(1)):
            u = int(connected_edges[0, i])
            v = int(connected_edges[1, i])
            # VGAE decoder uses inner product; apply sigmoid to map to (0,1)
            s = torch.sigmoid((z[u] * z[v]).sum()).item()
            preds[(u, v)] = float(s)
    return preds

#____________________________________________________________________________________________________________________________________________

def tr_remove_highest_n_edges_from_graph(data, predictions, n_edges, promo_mode):
    """
    Removes the top n edges with the highest or lowest predicted change position from the Citeseer graph.
    
    Args:
        data (torch_geometric.data.Data): Citeseer graph data.
        predictions (dict): Sorted predicted change positions.
        n_edges (int): Number of edges to remove.
        promo_mode (bool): If True, removes edges with the lowest predicted change position instead.
     
    Returns:
        torch_geometric.data.Data, list: Updated graph with edges removed and a list of removed edges.
    """
    edges_sorted = sorted(predictions.keys(), key=lambda x: predictions[x], reverse=not promo_mode)
    edges_to_remove = set(edges_sorted[:n_edges])
    mask = torch.tensor([tuple(edge.tolist()) not in edges_to_remove for edge in data.edge_index.t()], dtype=torch.bool)
    
    updated_data = data.clone()
    updated_data.edge_index = data.edge_index[:, mask]
    
    removed_edges = list(edges_to_remove)
    
    return removed_edges

#____________________________________________________________________________________________________________________________________________

def build_link_prediction_attack(data, selected_nodes, vgae_model, vgae_embedding, budget, promo_mode): 
    
    test_nodes =  selected_nodes
    test_nodes_int = test_nodes.item()

    predictions = predict_edges_vgae(vgae_model, vgae_embedding, data, test_nodes_int)
    edges_to_remove = tr_remove_highest_n_edges_from_graph(data, predictions, budget, promo_mode)
    updated_data = gold_remove_edges(data, edges_to_remove, budget)
 

    return updated_data

#____________________________________________________________________________________________________________________________________________