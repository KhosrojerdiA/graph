
import torch
from torch_geometric.utils import remove_self_loops
from torch_geometric.transforms import NormalizeFeatures
from utils.utils import gold_remove_edges, create_test_set_trained_model

#____________________________________________________________________________________________________________________________________________

import torch
from torch_geometric.transforms import NormalizeFeatures

def predict_rl(model, test_set, data, n_edges, promo_mode):
    """
    Predict edges to remove using the trained RL-SV2 model.
    
    Args:
        model (RLAgent): Trained RL model.
        test_set (Data): Graph data for testing.
        data (Data): Full graph data.
        n_edges (int): Number of edges to remove.
        promo_mode (bool): If True, removes edges with the highest scores; otherwise, removes the lowest.

    Returns:
        List of edges to remove (tuples of node pairs).
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_set = NormalizeFeatures()(test_set).to(device)
    data = NormalizeFeatures()(data).to(device)
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        # Get node scores
        node_scores = model(test_set.x)
        
        # Compute edge scores as the average of source and target node scores
        src_nodes = test_set.edge_index[0]
        dst_nodes = test_set.edge_index[1]
        
        edge_scores = (node_scores[src_nodes] + node_scores[dst_nodes]) / 2  # Edge score based on node importance

        # Select edges based on pruning mode
        if promo_mode:
            sorted_indices = torch.argsort(edge_scores, descending=True)[:n_edges]  # Highest scores
        else:
            sorted_indices = torch.argsort(edge_scores)[:n_edges]  # Lowest scores

        if n_edges == 1:
            sorted_indices = sorted_indices[0].unsqueeze(0)  # Ensure a single edge is handled correctly
        
        # Get edges to remove
        edges_to_remove = test_set.edge_index[:, sorted_indices]

        # Convert to a list of tuples
        edges_to_remove_list = list(map(tuple, edges_to_remove.t().tolist()))  # Format [(node1, node2), ...]

    return edges_to_remove_list


#____________________________________________________________________________________________________________________________________________

def build_rl_attack(data, selected_nodes, rlp_model, budget, promo_mode):
    
    test_set = create_test_set_trained_model(data, selected_nodes)
    edges_to_remove = predict_rl(rlp_model, test_set, data, budget, promo_mode)
    updated_data = gold_remove_edges(data, edges_to_remove, budget)
    #print(f"___________________________________")
    #print(f"Starting to remove!")
    #print("Removed: ", edges_to_remove)
    #print(f"Removing completed!")
    #print(f"___________________________________")

    return updated_data

#____________________________________________________________________________________________________________________________________________