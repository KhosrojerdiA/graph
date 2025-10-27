import torch
import random
from torch_geometric.utils import to_networkx, remove_self_loops
import networkx as nx

#If promo_mode is False, 
#the function will remove edges connected to the selected nodes that have the highest PageRank scores

#____________________________________________________________________________________________________________________________________________

def build_page_rank_attack(data, selected_nodes, budget, promo_mode): 
    
    if isinstance(selected_nodes, torch.Tensor) and selected_nodes.ndimension() == 0:
        selected_nodes = selected_nodes.unsqueeze(0)
    
    selected_nodes = selected_nodes.tolist()
    edge_index = data.edge_index.clone()
    edges = edge_index.t().tolist()
    
    G = to_networkx(data.to('cpu'), to_undirected=True)
    pagerank_scores = nx.pagerank(G)
    removed_edges_count = 0
    
    while removed_edges_count < budget and edges:
        node = random.choice(selected_nodes)
        node_edges = [(i, (src, dst)) for i, (src, dst) in enumerate(edges) if src == node or dst == node]
        
        if node_edges:
            if promo_mode:
                # Remove edge with the lowest PageRank score
                edge_to_remove = min(node_edges, key=lambda x: pagerank_scores[x[1][1]] if x[1][0] == node else pagerank_scores[x[1][0]])
            else:
                # Remove edge with the highest PageRank score
                edge_to_remove = max(node_edges, key=lambda x: pagerank_scores[x[1][1]] if x[1][0] == node else pagerank_scores[x[1][0]])
            
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

def build_p_page_rank_attack(data, selected_nodes, budget, promo_mode): #personalized
    """
    Perform edge-removal attack based on personalized PageRank scores.

    Args:
        data (torch_geometric.data.Data): Input graph data.
        selected_nodes (torch.Tensor): Target nodes to personalize PageRank toward.
        budget (int): Number of edges to remove.
        promo_mode (bool): 
            If True → remove edges connected to nodes with lowest personalized PageRank (promotion).
            If False → remove edges connected to nodes with highest personalized PageRank (demotion).

    Returns:
        torch_geometric.data.Data: Graph with edges removed.
    """
    # --- Convert selected_nodes to list
    if isinstance(selected_nodes, torch.Tensor):
        if selected_nodes.ndimension() == 0:
            selected_nodes = selected_nodes.unsqueeze(0)
        selected_nodes = selected_nodes.tolist()

    # --- Copy edges
    edge_index = data.edge_index.clone()
    edges = edge_index.t().tolist()

    # --- Create undirected NetworkX graph
    G = to_networkx(data.to('cpu'), to_undirected=True)

    # --- Build personalization vector
    personalization = {node: 0 for node in G.nodes()}
    for node in selected_nodes:
        if node in personalization:
            personalization[node] = 1.0 / len(selected_nodes)

    # --- Compute personalized PageRank
    pagerank_scores = nx.pagerank(G, alpha=0.85, personalization=personalization)

    removed_edges_count = 0

    while removed_edges_count < budget and edges:
        # Randomly choose one node from the selected set
        node = random.choice(selected_nodes)

        # Find all edges connected to that node
        node_edges = [(i, (src, dst)) for i, (src, dst) in enumerate(edges) if src == node or dst == node]

        if node_edges:
            # Depending on promo_mode, remove edge to low/high PageRank neighbor
            if promo_mode:
                # Remove edge with lowest personalized PageRank score
                edge_to_remove = min(
                    node_edges,
                    key=lambda x: pagerank_scores[x[1][1]] if x[1][0] == node else pagerank_scores[x[1][0]]
                )
            else:
                # Remove edge with highest personalized PageRank score
                edge_to_remove = max(
                    node_edges,
                    key=lambda x: pagerank_scores[x[1][1]] if x[1][0] == node else pagerank_scores[x[1][0]]
                )

            edges.pop(edge_to_remove[0])
            removed_edges_count += 1

    # --- Build new edge_index tensor
    new_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    new_edge_index, _ = remove_self_loops(new_edge_index)

    # --- Create updated dataset
    updated_data = data.clone()
    updated_data.edge_index = new_edge_index

    #print(f"[Attack] Removed {removed_edges_count} edges using Personalized PageRank")
    return updated_data
#____________________________________________________________________________________________________________________________________________