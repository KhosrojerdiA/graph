 
import torch
import random
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import degree

#____________________________________________________________________________________________________________________________________________

def build_cluster_attack(data, selected_nodes, clusters, budget, promo_mode):
    """
    Removes edges based on cluster sizes. 
    
    - If promo_mode=True (promotion), remove edges connected to the largest cluster.
    - If promo_mode=False (demotion), remove edges connected to the smallest cluster.
    
    
    Args:
        data (torch_geometric.data.Data): The input graph.
        selected_nodes (torch.Tensor or list): Nodes to attack.
        budget (int): Number of edges to remove.
        promo_mode (bool): Whether to promote or demote nodes.
    
    Returns:
        torch_geometric.data.Data: Updated graph with edges removed.
    """
    if isinstance(selected_nodes, torch.Tensor) and selected_nodes.ndimension() == 0:
        selected_nodes = selected_nodes.unsqueeze(0)

    selected_nodes = selected_nodes.tolist()
    edge_index = data.edge_index.clone()
    edges = edge_index.t().tolist()
    
    degrees = degree(edge_index[0], num_nodes=data.num_nodes)
    #clusters = compute_clusters(data)

    # Compute cluster sizes
    cluster_sizes = {}
    for node, cluster in clusters.items():
        cluster_sizes[cluster] = cluster_sizes.get(cluster, 0) + 1

    removed_edges_count = 0

    while removed_edges_count < budget and edges:
        node = random.choice(selected_nodes)
        node_edges = [(i, (src, dst)) for i, (src, dst) in enumerate(edges) if src == node or dst == node]

        if node_edges:
            if not promo_mode: #Demotion
                # Find the node in the smallest cluster
                target_node = min(node_edges, key=lambda x: cluster_sizes[clusters[x[1][1]]] if x[1][0] == node else cluster_sizes[clusters[x[1][0]]])
            else: #Promotion
                # Find the node in the largest cluster
                target_node = max(node_edges, key=lambda x: cluster_sizes[clusters[x[1][1]]] if x[1][0] == node else cluster_sizes[clusters[x[1][0]]])

            edges.pop(target_node[0])
            removed_edges_count += 1

    # Create new edge_index tensor
    new_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    new_edge_index, _ = remove_self_loops(new_edge_index)

    # Assign the modified edge_index to create the updated dataset
    updated_data = data.clone()
    updated_data.edge_index = new_edge_index

    return updated_data

#____________________________________________________________________________________________________________________________________________