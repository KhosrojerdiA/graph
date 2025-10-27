
from utils.utils import gold_remove_edges

#____________________________________________________________________________________________________________________________________________


def gold_get_highest_change_edges(simplified_performance, node, n_edges, promo_mode):
    """
    Get the top-n edges of the given node that have the highest or lowest change_position.
    """
    if node not in simplified_performance:
        return []
    
    node_edges = simplified_performance[node]
    if not node_edges:
        return []
    
    # Sort edges by change position
    sorted_edges = sorted(node_edges.items(), key=lambda x: x[1], reverse=not promo_mode)
    highest_edges = [edge for edge, _ in sorted_edges[:n_edges]]
    
    return highest_edges


#____________________________________________________________________________________________________________________________________________


def build_gold_attack(data, simplified_performance, selected_nodes, budget, promo_mode):
    
    highest_edges = gold_get_highest_change_edges(simplified_performance, selected_nodes.item(), budget, promo_mode)
    updated_data = gold_remove_edges(data, highest_edges, budget)
 
    return updated_data

#____________________________________________________________________________________________________________________________________________
